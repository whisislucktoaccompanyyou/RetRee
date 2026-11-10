"""Runnable entry point for ReTree on local data.

The upstream scripts hard-code EyeQ paths, default to inference (which needs
author checkpoints that are not in the repo), and pair images with masks by
glob order. This script pairs DRIVE by id and trains the mask-conditioned
image DDPM (segmentation -> fundus), which is the step that turns a vessel
map back into a retinal image.

DRIVE layout expected:
    <drive>/training/images_png/*_training.png
    <drive>/training/1st_manual/*_manual1.gif
    <drive>/training/mask/*_training_mask.gif   (FOV, optional)

Usage (from this directory):
    CUDA_VISIBLE_DEVICES=0 python -u run_retree.py --epochs 2 --batch-size 2
"""

import argparse
import os
import re

import torch
import torch.nn as nn
from PIL import Image
from torch import optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF
from tqdm import tqdm

from DDPM_model import DDPM


def save_checkpoint(model, filename):
    # Weights only, written to one rolling file. Keeping every epoch filled the
    # disk: each file was about 1.6 GB once the optimizer state was included.
    tmp = filename + ".tmp"
    torch.save({"state_dict": model.state_dict()}, tmp)
    os.replace(tmp, filename)


def save_images(images, path):
    grid = torch.cat([img for img in images.cpu()], dim=-1)
    array = grid.permute(1, 2, 0).numpy()
    if array.shape[-1] == 1:
        array = array[:, :, 0]
    Image.fromarray(array).save(path)


class Diffusion:
    """Cosine noise schedule, copied from train_ddpm_image.Diffusion so this
    entry point does not import utils' unused albumentations dependency."""

    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=256, device="cuda"):
        self.noise_steps = noise_steps
        self.img_size = img_size
        self.device = device
        beta = torch.abs(torch.cos(torch.linspace(0, torch.pi / 2, noise_steps)) * beta_end - (beta_end - beta_start))
        self.beta = beta.to(device)
        self.alpha = 1.0 - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def noise_images(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        eps = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * eps, eps

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))


def pad_to_square(image, fill=0):
    """Pad the short side with black. Resize() alone stretches a non-square fundus into an ellipse."""
    width, height = image.size
    side = max(width, height)
    if width == height:
        return image
    canvas = Image.new(image.mode, (side, side), fill)
    canvas.paste(image, ((side - width) // 2, (side - height) // 2))
    return canvas


def pair_id(name):
    match = re.match(r"(\d+)", os.path.basename(name))
    if match is None:
        raise ValueError(f"cannot parse DRIVE id from {name}")
    return match.group(1)


def collect_drive(root, split):
    split_dir = os.path.join(root, split)
    image_dir = os.path.join(split_dir, "images_png")
    if not os.path.isdir(image_dir):
        image_dir = os.path.join(split_dir, "images")
    vessel_dir = os.path.join(split_dir, "1st_manual")
    fov_dir = os.path.join(split_dir, "mask")

    images = {}
    for sub in ("images", "images_png"):
        folder = os.path.join(split_dir, sub)
        if not os.path.isdir(folder):
            continue
        for name in os.listdir(folder):
            if name.startswith("."):
                continue
            key = pair_id(name)
            # png is already decoded; prefer it over the original tif
            if key not in images or name.lower().endswith(".png"):
                images[key] = os.path.join(folder, name)

    vessels = {}
    if os.path.isdir(vessel_dir):
        for name in os.listdir(vessel_dir):
            if name.startswith("."):
                continue
            vessels[pair_id(name)] = os.path.join(vessel_dir, name)

    fovs = {}
    if os.path.isdir(fov_dir):
        for name in os.listdir(fov_dir):
            if name.startswith("."):
                continue
            fovs[pair_id(name)] = os.path.join(fov_dir, name)

    missing = sorted(set(images) - set(vessels))
    if missing:
        raise FileNotFoundError(f"{split}: images without a vessel mask: {missing}")

    pairs = []
    for key in sorted(images):
        pairs.append((images[key], vessels[key], fovs.get(key)))
    if not pairs:
        raise FileNotFoundError(f"no images under {image_dir}")
    return pairs


class PairedDrive(Dataset):
    """Fundus image (3ch) paired with its vessel map (1ch), both in [-1, 1]."""

    def __init__(self, pairs, img_size):
        self.pairs = pairs
        self.image_tf = transforms.Compose([
            transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BICUBIC),
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
        image_path, vessel_path, fov_path = self.pairs[index]
        image = Image.open(image_path).convert("RGB")
        vessel = Image.open(vessel_path).convert("L")
        if fov_path is not None:
            fov = Image.open(fov_path).convert("L").resize(image.size, Image.NEAREST)
            image = Image.composite(image, Image.new("RGB", image.size), fov)
        image = pad_to_square(image, 0)
        vessel = pad_to_square(vessel, 0)

        image = self.image_tf(image)
        vessel = self.mask_tf(vessel)
        if torch.rand(1).item() < 0.5:
            image, vessel = TF.hflip(image), TF.hflip(vessel)
        if torch.rand(1).item() < 0.5:
            image, vessel = TF.vflip(image), TF.vflip(vessel)
        return image, vessel


def sample_and_save(diffusion, model, vessels, out_dir, epoch, step, sample_steps):
    model.eval()
    n = vessels.shape[0]
    # Preview only. Stride the trained schedule so a checkpoint image does not
    # occupy the GPU for a full 1000-step sample.
    timesteps = torch.linspace(diffusion.noise_steps - 1, 1, sample_steps).long()
    with torch.no_grad():
        x = torch.randn((n, 3, diffusion.img_size, diffusion.img_size), device=vessels.device)
        for index, i in enumerate(timesteps.tolist()):
            t = torch.full((n,), i, device=vessels.device, dtype=torch.long)
            predicted = model(x, t, vessels)
            alpha = diffusion.alpha[t][:, None, None, None]
            alpha_hat = diffusion.alpha_hat[t][:, None, None, None]
            beta = diffusion.beta[t][:, None, None, None]
            noise = torch.randn_like(x) if index < len(timesteps) - 1 else torch.zeros_like(x)
            x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / torch.sqrt(1 - alpha_hat)) * predicted) + torch.sqrt(beta) * noise
    model.train()

    fake = ((x.clamp(-1, 1) + 1) / 2 * 255).type(torch.uint8)
    label = ((vessels.clamp(-1, 1) + 1) / 2 * 255).type(torch.uint8)
    save_images(fake, os.path.join(out_dir, f"epoch{epoch}_step{step}_fake.png"))
    save_images(label, os.path.join(out_dir, f"epoch{epoch}_step{step}_mask.png"))


def train(args):
    device = torch.device(args.device)
    pairs = collect_drive(args.drive, "training")
    print(f"paired {len(pairs)} DRIVE training images")

    loader = DataLoader(
        PairedDrive(pairs, args.image_size),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=len(pairs) > args.batch_size,
    )

    os.makedirs(args.out, exist_ok=True)
    ckpt_dir = os.path.join(args.out, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    model = DDPM(img_channels=4, time_dim=args.emb_dim).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    diffusion = Diffusion(noise_steps=args.noise_steps, img_size=args.image_size, device=device)
    mse = nn.MSELoss()
    l1 = nn.L1Loss()

    for epoch in range(1, args.epochs + 1):
        running = 0.0
        pbar = tqdm(loader, desc=f"epoch {epoch}/{args.epochs}")
        for step, (images, vessels) in enumerate(pbar, start=1):
            images = images.to(device)
            vessels = vessels.to(device)
            t = diffusion.sample_timesteps(images.shape[0]).to(device)
            x_t, noise = diffusion.noise_images(images, t)

            predicted = model(x_t, t, vessels)
            loss = mse(predicted, noise) + l1(predicted, noise)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            running += loss.item()
            pbar.set_postfix(loss=f"{running / step:.4f}")

            if step == len(loader):
                sample_and_save(diffusion, model, vessels[: args.sample_n], args.out, epoch, step, args.sample_steps)

        save_checkpoint(model, os.path.join(ckpt_dir, "ddpm_latest.pth.tar"))
        print(f"epoch {epoch} avg loss {running / max(len(loader), 1):.4f}")


def parse_args():
    here = os.path.dirname(os.path.abspath(__file__))
    default_drive = os.path.abspath(os.path.join(here, "..", "data", "DRIVE"))
    parser = argparse.ArgumentParser(description="Train ReTree mask->image DDPM on DRIVE")
    parser.add_argument("--drive", default=default_drive)
    parser.add_argument("--out", default=os.path.join(here, "results", "drive_seg2img"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--emb-dim", type=int, default=256)
    parser.add_argument("--noise-steps", type=int, default=1000)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--sample-n", type=int, default=2)
    parser.add_argument("--sample-steps", type=int, default=50)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
