"""Turn vessel masks back into fundus images with the trained ReTree DDPM.

Reads paired files from a flat directory:
    <id>_mask.png    conditioning vessel map
    <id>_image.png   original fundus image, kept only for the comparison strip

Writes one strip per id to the output directory: original | mask | generated.
"""

import argparse
import os
import re

import torch
from PIL import Image
from torchvision import transforms

from DDPM_model import DDPM
from run_retree import Diffusion


def pair_id(name):
    match = re.match(r"(\d+)", name)
    return match.group(1) if match else None


def collect_pairs(folder):
    masks = {}
    images = {}
    for name in os.listdir(folder):
        key = pair_id(name)
        if key is None:
            continue
        path = os.path.join(folder, name)
        lower = name.lower()
        if "mask" in lower:
            masks[key] = path
        elif "image" in lower:
            images[key] = path
    missing = sorted(set(masks) - set(images))
    if missing:
        raise FileNotFoundError(f"masks without a matching image: {missing}")
    if not masks:
        raise FileNotFoundError(f"no *_mask.png under {folder}")
    return [(key, images[key], masks[key]) for key in sorted(masks)]


def load_mask(path, img_size):
    mask = Image.open(path).convert("L")
    return transforms.Compose([
        transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.NEAREST),
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5),
    ])(mask)


def generate(model, diffusion, mask, steps):
    # The reverse step assumes t decreases by exactly 1. Subsampling the trained
    # 1000-step schedule leaves the update inconsistent, which samples pure noise.
    if steps != diffusion.noise_steps:
        raise ValueError(
            f"--sample-steps must equal --noise-steps ({diffusion.noise_steps}); "
            "this sampler cannot skip steps"
        )
    x = torch.randn((1, 3, diffusion.img_size, diffusion.img_size), device=mask.device)
    for i in range(diffusion.noise_steps - 1, 0, -1):
        t = torch.full((1,), i, device=mask.device, dtype=torch.long)
        predicted = model(x, t, mask)
        alpha = diffusion.alpha[t][:, None, None, None]
        alpha_hat = diffusion.alpha_hat[t][:, None, None, None]
        beta = diffusion.beta[t][:, None, None, None]
        noise = torch.randn_like(x) if i > 1 else torch.zeros_like(x)
        x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / torch.sqrt(1 - alpha_hat)) * predicted) + torch.sqrt(beta) * noise
    return ((x.clamp(-1, 1) + 1) / 2 * 255).type(torch.uint8)[0]


def side_by_side(original_path, mask_path, generated, img_size):
    original = Image.open(original_path).convert("RGB").resize((img_size, img_size), Image.BICUBIC)
    mask = Image.open(mask_path).convert("RGB").resize((img_size, img_size), Image.NEAREST)
    fake = Image.fromarray(generated.permute(1, 2, 0).cpu().numpy())
    strip = Image.new("RGB", (img_size * 3, img_size))
    strip.paste(original, (0, 0))
    strip.paste(mask, (img_size, 0))
    strip.paste(fake, (img_size * 2, 0))
    return strip


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description="ReTree mask -> fundus inference")
    parser.add_argument("--input", default=os.path.join(here, "test_input"))
    parser.add_argument("--output", default=os.path.join(here, "test_output"))
    parser.add_argument("--checkpoint", default=os.path.join(
        here, "results", "drive_seg2img", "checkpoints", "ddpm_latest.pth.tar"))
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--emb-dim", type=int, default=256)
    parser.add_argument("--noise-steps", type=int, default=1000)
    parser.add_argument("--sample-steps", type=int, default=1000)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)
    pairs = collect_pairs(args.input)
    print(f"{len(pairs)} pairs")

    model = DDPM(img_channels=4, time_dim=args.emb_dim).to(device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state["state_dict"])
    model.eval()
    diffusion = Diffusion(noise_steps=args.noise_steps, img_size=args.image_size, device=device)

    os.makedirs(args.output, exist_ok=True)
    for key, image_path, mask_path in pairs:
        mask = load_mask(mask_path, args.image_size).unsqueeze(0).to(device)
        with torch.no_grad():
            generated = generate(model, diffusion, mask, args.sample_steps)
        side_by_side(image_path, mask_path, generated, args.image_size).save(
            os.path.join(args.output, f"{key}_compare.png"))
        Image.fromarray(generated.permute(1, 2, 0).cpu().numpy()).save(
            os.path.join(args.output, f"{key}_generated.png"))
        print(f"wrote {key}")


if __name__ == "__main__":
    main()
