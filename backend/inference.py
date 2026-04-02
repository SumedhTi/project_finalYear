import torch
import numpy as np
import cv2
from basicsr.models.archs.NAFNet_arch import NAFNet

device = torch.device("cpu")  # force CPU

def load_model(weights_path):
    ckpt = torch.load(weights_path, map_location="cpu")
    state_dict = ckpt.get('params', ckpt.get('state_dict', ckpt))

    # --- Infer architecture dynamically ---
    # Count encoder blocks
    enc_blk_nums = []
    i = 0
    while True:
        key = f"encoders.{i}.0.conv1.weight"
        if key in state_dict:
            count = 0
            while f"encoders.{i}.{count}.conv1.weight" in state_dict:
                count += 1
            enc_blk_nums.append(count)
            i += 1
        else:
            break

    # Count decoder blocks
    dec_blk_nums = []
    i = 0
    while True:
        key = f"decoders.{i}.0.conv1.weight"
        if key in state_dict:
            count = 0
            while f"decoders.{i}.{count}.conv1.weight" in state_dict:
                count += 1
            dec_blk_nums.append(count)
            i += 1
        else:
            break

    # Middle blocks
    middle_blk_num = 0
    while f"middle_blks.{middle_blk_num}.conv1.weight" in state_dict:
        middle_blk_num += 1

    # Width (channels)
    width = state_dict["intro.weight"].shape[0]

    # --- Build model ---
    model = NAFNet(
        width=width,
        enc_blk_nums=enc_blk_nums,
        middle_blk_num=middle_blk_num,
        dec_blk_nums=dec_blk_nums
    )

    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    return model

def resize_image(img, max_size):
    h, w = img.shape[:2]

    # Scale factor
    scale = min(max_size / max(h, w), 1.0)

    new_h = int(h * scale)
    new_w = int(w * scale)

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    return resized, (h, w)  # return original size


def preprocess(img, size):
    img, original_size = resize_image(img, size)

    img = img.astype(np.float32) / 255.0

    # Ensure divisible by 8
    h, w = img.shape[:2]
    h = h - h % 8
    w = w - w % 8
    img = img[:h, :w]

    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)

    return img.to(device), original_size


def postprocess(tensor, original_size):
    img = tensor.squeeze().permute(1, 2, 0).cpu().detach().numpy()
    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)

    # Resize back to original resolution
    img = cv2.resize(img, (original_size[1], original_size[0]), interpolation=cv2.INTER_LINEAR)

    return img


@torch.no_grad()
def run_pipeline(img, deblur_model, denoise_model, size, mode):
    x, original_size = preprocess(img, size)
    if mode == "Deblur":
        x = deblur_model(x)
    elif mode == "Denoise":
        x = denoise_model(x)
    else:
        x = deblur_model(x)
        x = denoise_model(x) 

    out = postprocess(x, original_size)
    return out
