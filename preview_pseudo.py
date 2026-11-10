"""Preview the segmenter on a few EyeQ Good images. Does not train anything.

Writes overlay strips to results/retree/pseudo_preview/. The old pseudo_images
and pseudo_masks directories are left untouched.
"""

import argparse
import os

import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional as TF

from run_retree import pad_to_square
from train_pipeline import Segmenter, good_eyeq


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    data = os.path.abspath(os.path.join(here, "..", "data"))
    parser = argparse.ArgumentParser(description="Preview vessel pseudo labels")
    parser.add_argument("--kaggle", default=os.path.join(data, "kaggle", "extracted", "train"))
    parser.add_argument("--quality-csv", default=os.path.join(data, "kaggle", "Label_EyeQ_train.csv"))
    parser.add_argument("--checkpoint", default=os.path.join(here, "results", "retree", "segmenter.pth.tar"))
    parser.add_argument("--out", default=os.path.join(here, "results", "retree", "pseudo_preview"))
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    keep = good_eyeq(args.quality_csv)
    names = sorted(name for name in os.listdir(args.kaggle) if name in keep)[: args.n]
    if not names:
        raise FileNotFoundError(f"no EyeQ Good images in {args.kaggle}")
    print(f"preview {len(names)} images at {args.image_size}")

    device = torch.device(args.device)
    model = Segmenter().to(device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state["state_dict"])
    model.eval()

    to_tensor = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
    ])
    os.makedirs(args.out, exist_ok=True)

    with torch.no_grad():
        for name in names:
            image = pad_to_square(Image.open(os.path.join(args.kaggle, name)).convert("RGB"), 0)
            tensor = to_tensor(image).unsqueeze(0).to(device)
            mask = (torch.sigmoid(model(tensor)) > 0.5).float()[0]
            shown = TF.to_pil_image(tensor[0].cpu())
            mask_img = TF.to_pil_image(mask.cpu())
            red = Image.new("RGB", shown.size, (255, 0, 0))
            overlay = Image.composite(red, shown, mask_img)
            strip = Image.new("RGB", (args.image_size * 3, args.image_size))
            strip.paste(shown, (0, 0))
            strip.paste(mask_img.convert("RGB"), (args.image_size, 0))
            strip.paste(overlay, (args.image_size * 2, 0))
            stem = os.path.splitext(name)[0]
            strip.save(os.path.join(args.out, stem + "_overlay.png"))
            print(f"wrote {stem} vessel_pixels {int(mask.sum().item())}", flush=True)


if __name__ == "__main__":
    main()
