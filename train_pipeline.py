"""ReTree training on DRIVE plus the Kaggle EyePACS images.

The Kaggle set has fundus images but no vessel maps, so it cannot supervise
the mask-to-image model on its own. The paper's own steps are reproduced here:

  1. Train a vessel segmenter on DRIVE, the only paired labels available.
  2. Segment every Kaggle image. Those masks are the pseudo labels.
  3. Train the mask-conditioned image DDPM on DRIVE pairs plus the pseudo pairs.

Checkpoints are weights only and one rolling file per stage, so a long run
cannot fill the disk the way per-epoch optimizer dumps did.
"""

import argparse
import csv
import os
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch import optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF
from tqdm import tqdm

from run_retree import Diffusion, collect_drive, pad_to_square, save_checkpoint


class Segmenter(nn.Module):
    """Small U-Net. DRIVE only has 20 labelled images, so a large one overfits."""

    def __init__(self, base=32):
        super().__init__()

        def block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )

        self.enc1 = block(3, base)
        self.enc2 = block(base, base * 2)
        self.enc3 = block(base * 2, base * 4)
        self.pool = nn.MaxPool2d(2)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = block(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = block(base * 2, base)
        self.head = nn.Conv2d(base, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        d2 = self.dec2(torch.cat([self.up2(e3), e2], 1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], 1))
        return self.head(d1)


class DriveSeg(Dataset):
    def __init__(self, pairs, img_size, augment, patches=1, patch_size=0):
        self.pairs = pairs
        self.augment = augment
        self.patches = patches
        self.patch_size = patch_size
        self.image_tf = transforms.Compose([
            transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
        ])
        self.mask_tf = transforms.Compose([
            transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.pairs) * self.patches

    def __getitem__(self, index):
        image_path, vessel_path, fov_path = self.pairs[index % len(self.pairs)]
        image = Image.open(image_path).convert("RGB")
        vessel = Image.open(vessel_path).convert("L")
        if fov_path is not None:
            fov = Image.open(fov_path).convert("L").resize(image.size, Image.NEAREST)
            image = Image.composite(image, Image.new("RGB", image.size), fov)
        image = pad_to_square(image, 0)
        vessel = pad_to_square(vessel, 0)
        image = self.image_tf(image)
        vessel = (self.mask_tf(vessel) > 0.5).float()
        if self.augment:
            if torch.rand(1).item() < 0.5:
                image, vessel = TF.hflip(image), TF.hflip(vessel)
            if torch.rand(1).item() < 0.5:
                image, vessel = TF.vflip(image), TF.vflip(vessel)
            angle = random.uniform(-180, 180)
            image = TF.rotate(image, angle)
            vessel = TF.rotate(vessel, angle)
        if self.patch_size and self.patch_size < image.shape[-1]:
            top = torch.randint(0, image.shape[-2] - self.patch_size + 1, ()).item()
            left = torch.randint(0, image.shape[-1] - self.patch_size + 1, ()).item()
            image = image[:, top:top + self.patch_size, left:left + self.patch_size]
            vessel = vessel[:, top:top + self.patch_size, left:left + self.patch_size]
        return image, vessel


def good_eyeq(csv_path):
    """EyeQ quality is a grade, not a score: 0 Good, 1 Usable, 2 Reject. Higher is worse."""
    counts = {}
    keep = set()
    with open(csv_path, newline="") as handle:
        for row in csv.DictReader(handle):
            grade = int(row["quality"])
            counts[grade] = counts.get(grade, 0) + 1
            if grade == 0:
                keep.add(row["image"])
    print(
        "EyeQ grades: "
        f"Good(0)={counts.get(0, 0)} "
        f"Usable(1)={counts.get(1, 0)} "
        f"Reject(2)={counts.get(2, 0)}; "
        f"keeping Good only, {len(keep)} names"
    )
    return keep


def collect_stare(image_dir, label_dir):
    """Pair STARE images with Hoover's .ah vessel maps. Both are 700x605 PPM."""
    labels = {}
    if not os.path.isdir(label_dir):
        raise FileNotFoundError(f"STARE labels not found: {label_dir}")
    for name in os.listdir(label_dir):
        if not name.endswith(".ah.ppm"):
            continue
        labels[name[: -len(".ah.ppm")]] = os.path.join(label_dir, name)
    pairs = []
    missing = []
    for key in sorted(labels):
        image = os.path.join(image_dir, key + ".ppm")
        if os.path.isfile(image):
            pairs.append((image, labels[key], None))
        else:
            missing.append(key)
    if missing:
        raise FileNotFoundError(f"STARE labels without an image: {missing}")
    if not pairs:
        raise FileNotFoundError(f"no .ah.ppm labels in {label_dir}")
    return pairs


def train_segmenter(args, device):
    drive_pairs = collect_drive(args.drive, "training")
    stare_pairs = collect_stare(args.stare, args.stare_labels)
    # DRIVE test ships no vessel labels. Hold out the tail of each labelled set
    # so the score is not only the dataset the model just trained on.
    drive_at = max(len(drive_pairs) - 4, 1)
    stare_at = max(len(stare_pairs) - 4, 1)
    train_pairs = drive_pairs[:drive_at] + stare_pairs[:stare_at]
    test_pairs = drive_pairs[drive_at:] + stare_pairs[stare_at:]
    print(
        f"segmenter: DRIVE {drive_at}/{len(drive_pairs) - drive_at}, "
        f"STARE {stare_at}/{len(stare_pairs) - stare_at}, "
        f"total {len(train_pairs)} train, {len(test_pairs)} held-out"
    )

    train_set = DriveSeg(
        train_pairs, args.image_size, augment=True,
        patches=args.seg_patches, patch_size=args.seg_patch_size,
    )
    print(f"segmenter crops: {len(train_set)} patches/epoch, patch {args.seg_patch_size or args.image_size}")
    train_loader = DataLoader(
        train_set, batch_size=args.seg_batch_size, shuffle=True, num_workers=args.num_workers,
    )
    # Held-out score stays on full images, so it is comparable with the earlier runs.
    test_loader = DataLoader(
        DriveSeg(test_pairs, args.image_size, augment=False),
        batch_size=2, shuffle=False, num_workers=args.num_workers,
    )

    model = Segmenter().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([8.0], device=device))
    best_dice = -1.0
    ckpt = os.path.join(args.out, "segmenter.pth.tar")

    for epoch in range(1, args.seg_epochs + 1):
        model.train()
        running = 0.0
        for images, vessels in tqdm(train_loader, desc=f"seg {epoch}/{args.seg_epochs}"):
            images, vessels = images.to(device), vessels.to(device)
            loss = bce(model(images), vessels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            running += loss.item()

        dice = evaluate_segmenter(model, test_loader, device)
        print(f"seg epoch {epoch} loss {running / len(train_loader):.4f} test dice {dice:.4f}")
        if dice > best_dice:
            best_dice = dice
            save_checkpoint(model, ckpt)
            print(f"segmenter saved, dice {dice:.4f}")

    state = torch.load(ckpt, map_location=device, weights_only=False)
    model.load_state_dict(state["state_dict"])
    model.eval()
    return model


@torch.no_grad()
def evaluate_segmenter(model, loader, device):
    model.eval()
    intersection = 0.0
    total = 0.0
    for images, vessels in loader:
        pred = torch.sigmoid(model(images.to(device))) > 0.5
        target = vessels.to(device) > 0.5
        intersection += (pred & target).sum().item()
        total += pred.sum().item() + target.sum().item()
    return 2 * intersection / max(total, 1.0)


@torch.no_grad()
def write_pseudo_labels(model, args, device):
    keep = good_eyeq(args.quality_csv)
    names = sorted(name for name in os.listdir(args.kaggle) if name in keep)
    if not names:
        raise FileNotFoundError(f"no EyeQ Good (quality=0) images in {args.kaggle}")
    names = names[: args.max_images]
    print(f"pseudo labels: {len(names)} images")

    mask_dir = os.path.join(args.out, "pseudo_masks")
    image_dir = os.path.join(args.out, "pseudo_images")
    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(image_dir, exist_ok=True)
    to_tensor = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
    ])

    for start in tqdm(range(0, len(names), args.batch_size), desc="segment kaggle"):
        batch = names[start:start + args.batch_size]
        images = torch.stack([
            to_tensor(pad_to_square(Image.open(os.path.join(args.kaggle, name)).convert("RGB"), 0))
            for name in batch
        ]).to(device)
        masks = (torch.sigmoid(model(images)) > 0.5).float()
        for name, image, mask in zip(batch, images, masks):
            stem = os.path.splitext(name)[0]
            TF.to_pil_image(image.cpu()).save(os.path.join(image_dir, stem + ".png"))
            TF.to_pil_image(mask.cpu()).save(os.path.join(mask_dir, stem + ".png"))
    return image_dir, mask_dir


class PseudoPairs(Dataset):
    def __init__(self, image_dir, mask_dir, img_size):
        self.pairs = []
        for name in sorted(os.listdir(mask_dir)):
            image = os.path.join(image_dir, name)
            if os.path.exists(image):
                self.pairs.append((image, os.path.join(mask_dir, name)))
        if not self.pairs:
            raise FileNotFoundError(f"no pairs in {mask_dir}")
        self.image_tf = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        self.mask_tf = transforms.Compose([
            transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5),
        ])

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        image = self.image_tf(pad_to_square(Image.open(self.pairs[index][0]).convert("RGB"), 0))
        mask = self.mask_tf(pad_to_square(Image.open(self.pairs[index][1]).convert("L"), 0))
        if torch.rand(1).item() < 0.5:
            image, mask = TF.hflip(image), TF.hflip(mask)
        if torch.rand(1).item() < 0.5:
            image, mask = TF.vflip(image), TF.vflip(mask)
        return image, mask


def train_ddpm(args, image_dir, mask_dir, device):
    from DDPM_model import DDPM

    loader = DataLoader(
        PseudoPairs(image_dir, mask_dir, args.image_size),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, drop_last=True,
    )
    print(f"ddpm: {len(loader.dataset)} pairs, {len(loader)} steps/epoch")

    model = DDPM(img_channels=4, time_dim=args.emb_dim).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    diffusion = Diffusion(noise_steps=args.noise_steps, img_size=args.image_size, device=device)
    mse = nn.MSELoss()
    l1 = nn.L1Loss()
    ckpt = os.path.join(args.out, "ddpm_latest.pth.tar")

    for epoch in range(1, args.epochs + 1):
        running = 0.0
        pbar = tqdm(loader, desc=f"ddpm {epoch}/{args.epochs}")
        for step, (images, masks) in enumerate(pbar, start=1):
            images, masks = images.to(device), masks.to(device)
            t = diffusion.sample_timesteps(images.shape[0]).to(device)
            x_t, noise = diffusion.noise_images(images, t)
            predicted = model(x_t, t, masks)
            loss = mse(predicted, noise) + l1(predicted, noise)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            running += loss.item()
            pbar.set_postfix(loss=f"{running / step:.4f}")
        save_checkpoint(model, ckpt)
        print(f"ddpm epoch {epoch} avg loss {running / len(loader):.4f}", flush=True)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    data = os.path.abspath(os.path.join(here, "..", "data"))
    parser = argparse.ArgumentParser(description="ReTree pipeline: DRIVE segmenter, Kaggle pseudo labels, DDPM")
    parser.add_argument("--drive", default=os.path.join(data, "DRIVE"))
    parser.add_argument("--stare", default=os.path.join(data, "STARE"))
    parser.add_argument("--stare-labels", default=os.path.join(data, "labels-ah（stare的人工标注）"))
    parser.add_argument("--segmenter-only", action="store_true",
                        help="train the vessel segmenter and stop before pseudo labels and the DDPM")
    parser.add_argument("--kaggle", default=os.path.join(data, "kaggle", "extracted", "train"))
    parser.add_argument("--quality-csv", default=os.path.join(data, "kaggle", "Label_EyeQ_train.csv"))
    parser.add_argument("--max-images", type=int, default=4000)
    parser.add_argument("--out", default=os.path.join(here, "results", "retree"))
    parser.add_argument("--seg-epochs", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--seg-patch-size", type=int, default=0,
                        help="random crop size while training the segmenter; 0 uses the full image")
    parser.add_argument("--seg-patches", type=int, default=1,
                        help="random crops drawn from each labelled image per epoch")
    parser.add_argument("--seg-batch-size", type=int, default=4)
    parser.add_argument("--emb-dim", type=int, default=256)
    parser.add_argument("--noise-steps", type=int, default=1000)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = torch.device(args.device)
    segmenter = train_segmenter(args, device)
    if args.segmenter_only:
        print(f"segmenter only, saved to {os.path.join(args.out, 'segmenter.pth.tar')}")
        return
    image_dir, mask_dir = write_pseudo_labels(segmenter, args, device)
    del segmenter
    torch.cuda.empty_cache()
    train_ddpm(args, image_dir, mask_dir, device)


if __name__ == "__main__":
    main()
