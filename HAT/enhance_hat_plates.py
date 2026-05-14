"""
=====================================================================
  HAT Enhancement Script — Pakistan License Plates
  MPhil Thesis — Final Version

  INPUT  : HAT\datasets\LR_Enhance         (776x476 px images)
  OUTPUT : HAT\datasets\LR_Enhancement     (3104x1904 px enhanced)
=====================================================================
"""

import os
import sys
import cv2
import torch
import numpy as np
from pathlib import Path

# ─────────────────────────────────────────────────────────────
#  PATHS — do not change these
# ─────────────────────────────────────────────────────────────
BASE          = r"C:\Users\Sunny\PycharmProjects\Enhancement Model\HAT\datasets"
LR_INPUT      = os.path.join(BASE, "LR_Enhance")       # input  — never touched
LR_OUTPUT     = os.path.join(BASE, "LR_Enhancement")   # output — created automatically
MODEL_PATH    = r"experiments\pretrained_models\Real_HAT_GAN_SRx4.pth"

SCALE         = 4
WINDOW_SIZE   = 16
TILE_SIZE     = 256
TILE_PAD      = 32
# ─────────────────────────────────────────────────────────────

sys.path.insert(0, os.path.abspath('.'))


def check_model_file():
    if not os.path.exists(MODEL_PATH):
        print("\n" + "="*60)
        print("  ERROR: Pretrained weights not found!")
        print(f"  Expected: {os.path.abspath(MODEL_PATH)}")
        print("  Download: https://drive.google.com/file/d/1Ma12vCWT27P9M99-s2RXnynKN-OQsBrv/view")
        print("  Place in: experiments\\pretrained_models\\")
        print("="*60)
        sys.exit(1)


def get_image_files(folder):
    exts = {'.jpg', '.jpeg', '.png', '.bmp'}
    p = Path(folder)
    if not p.exists():
        print(f"\nERROR: Folder not found:\n  {folder}")
        sys.exit(1)
    files = sorted([f for f in p.iterdir() if f.suffix.lower() in exts])
    if not files:
        print(f"\nERROR: No images found in:\n  {folder}")
        sys.exit(1)
    return files


def load_model():
    print("\n[STEP 1/4]  Loading HAT model...")
    try:
        from hat.archs.hat_arch import HAT
    except ImportError as e:
        print(f"\nERROR: Cannot import HAT: {e}")
        print("Run: python setup.py develop")
        sys.exit(1)

    model = HAT(
        upscale=4, in_chans=3, img_size=64, window_size=16,
        compress_ratio=3, squeeze_factor=30, conv_scale=0.01,
        overlap_ratio=0.5, img_range=1.0,
        depths=[6,6,6,6,6,6], embed_dim=180,
        num_heads=[6,6,6,6,6,6], mlp_ratio=2,
        upsampler='pixelshuffle', resi_connection='1conv'
    )

    print(f"             Weights: {os.path.abspath(MODEL_PATH)}")
    ckpt = torch.load(MODEL_PATH, map_location='cpu')

    if 'params_ema' in ckpt:
        model.load_state_dict(ckpt['params_ema'], strict=True)
        print("             Key: params_ema  ✓")
    elif 'params' in ckpt:
        model.load_state_dict(ckpt['params'], strict=True)
        print("             Key: params  ✓")
    else:
        model.load_state_dict(ckpt, strict=True)
        print("             Key: direct  ✓")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = model.to(device)
    model.eval()

    if torch.cuda.is_available():
        name   = torch.cuda.get_device_name(0)
        vram   = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"             GPU: {name} ({vram:.1f} GB VRAM)  ✓")
    else:
        print("             CPU mode (no GPU found — will be slow)")

    print("             Model ready!  ✓")
    return model, device


def pad_to_multiple(tensor, window_size):
    """
    Pad image so H and W are multiples of window_size.
    HAT requires dimensions divisible by 16.
    Returns padded tensor + original H, W for cropping later.
    """
    _, _, h, w = tensor.shape
    pad_h = (window_size - h % window_size) % window_size
    pad_w = (window_size - w % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        tensor = torch.nn.functional.pad(
            tensor, (0, pad_w, 0, pad_h), mode='reflect'
        )
    return tensor, h, w


def tile_inference(model, tensor, tile_size, tile_pad, scale):
    """Process image in tiles to avoid GPU out-of-memory."""
    _, _, h, w = tensor.shape
    out_h = h * scale
    out_w = w * scale
    output  = torch.zeros((1, 3, out_h, out_w),
                           dtype=tensor.dtype,
                           device=tensor.device)

    tiles_x = max(1, -(-w // tile_size))
    tiles_y = max(1, -(-h // tile_size))

    for ty in range(tiles_y):
        for tx in range(tiles_x):
            x1 = max(tx * tile_size - tile_pad, 0)
            y1 = max(ty * tile_size - tile_pad, 0)
            x2 = min(tx * tile_size + tile_size + tile_pad, w)
            y2 = min(ty * tile_size + tile_size + tile_pad, h)

            tile_in  = tensor[:, :, y1:y2, x1:x2]
            tile_out = model(tile_in)

            dest_x1 = tx * tile_size * scale
            dest_y1 = ty * tile_size * scale
            dest_x2 = min((tx * tile_size + tile_size) * scale, out_w)
            dest_y2 = min((ty * tile_size + tile_size) * scale, out_h)

            src_x1  = (tx * tile_size - x1) * scale
            src_y1  = (ty * tile_size - y1) * scale
            src_x2  = src_x1 + (dest_x2 - dest_x1)
            src_y2  = src_y1 + (dest_y2 - dest_y1)

            output[:, :, dest_y1:dest_y2, dest_x1:dest_x2] = \
                tile_out[:, :, src_y1:src_y2, src_x1:src_x2]

    return output


def enhance_image(model, device, image_path):
    """Read → pad → enhance → crop → return."""
    img_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError(f"Cannot read: {image_path.name}")

    orig_h, orig_w = img_bgr.shape[:2]

    # BGR → RGB → float [0,1] → tensor (1,C,H,W)
    img_rgb   = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_float = img_rgb.astype(np.float32) / 255.0
    tensor    = torch.from_numpy(
        np.transpose(img_float, (2, 0, 1))
    ).unsqueeze(0).float().to(device)

    # Pad to multiple of 16 (HAT requirement)
    tensor, orig_h_t, orig_w_t = pad_to_multiple(tensor, WINDOW_SIZE)

    # Run HAT
    with torch.no_grad():
        out = tile_inference(model, tensor, TILE_SIZE, TILE_PAD, SCALE)

    # Crop padding from output
    out = out[:, :, :orig_h * SCALE, :orig_w * SCALE]

    # Tensor → numpy → BGR uint8
    out_np  = out.squeeze(0).cpu().float().clamp(0, 1).numpy()
    out_np  = np.transpose(out_np, (1, 2, 0))
    out_np  = (out_np * 255.0).round().astype(np.uint8)
    out_bgr = cv2.cvtColor(out_np, cv2.COLOR_RGB2BGR)

    return out_bgr, orig_w, orig_h


def main():
    print("=" * 65)
    print("  HAT — Hybrid Attention Transformer (TPAMI 2025)")
    print("  Pakistan License Plate Super-Resolution")
    print("  MPhil Thesis")
    print("=" * 65)

    # Checks
    check_model_file()
    lr_files = get_image_files(LR_INPUT)
    print(f"\n  Found {len(lr_files)} images in LR_Enhance folder.")

    # Create output folder automatically
    Path(LR_OUTPUT).mkdir(parents=True, exist_ok=True)
    print(f"\n[STEP 2/4]  Output folder ready:")
    print(f"             {LR_OUTPUT}  ✓")

    # Load model
    model, device = load_model()

    # Enhance
    print(f"\n[STEP 3/4]  Enhancing {len(lr_files)} images ({SCALE}x upscale)...")
    print(f"             Input  → {LR_INPUT}")
    print(f"             Output → {LR_OUTPUT}\n")

    success = 0
    failed  = 0

    for i, lr_file in enumerate(lr_files, 1):
        try:
            print(f"  [{i:>5}/{len(lr_files)}]  {lr_file.name:<25}", end="  ")
            enhanced, orig_w, orig_h = enhance_image(model, device, lr_file)
            enh_h, enh_w = enhanced.shape[:2]
            cv2.imwrite(str(Path(LR_OUTPUT) / lr_file.name), enhanced)
            print(f"{orig_w}x{orig_h} → {enh_w}x{enh_h}  ✓")
            success += 1
        except Exception as e:
            print(f"FAILED  ✗  ({e})")
            failed += 1

    # Summary
    print("\n" + "=" * 65)
    print(f"[STEP 4/4]  ALL DONE!")
    print(f"\n  Enhanced : {success} images  ✓")
    if failed:
        print(f"  Failed   : {failed} images  ✗")
    print(f"\n  Output saved to:")
    print(f"  {LR_OUTPUT}")
    print("\n  LR_Enhance folder was NOT touched.")
    print("=" * 65)


if __name__ == '__main__':
    main()
