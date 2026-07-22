import torch
import cv2
import numpy as np
import torch.nn as nn


def get_CPFA(img, pattern):
    """
    Generate a CPD mask based on the mosaic pattern and polarization angles.

    Args:
        img (torch.Tensor): Input image tensor of shape [b, c, h, w] (already on GPU).
        pattern (str): Mosaic pattern (e.g., 'rggb', 'grbg').

    Returns:
        mask (torch.Tensor): A tensor of shape [b, h, w, 12] representing the CPD mask.
    """
    # Step 1: Parse the mosaic pattern to determine channel positions
    pattern = pattern.lower()
    pr = [i for i, c in enumerate(pattern) if c == 'r']
    pg = [i for i, c in enumerate(pattern) if c == 'g']
    pb = [i for i, c in enumerate(pattern) if c == 'b']

    # Ensure the pattern has exactly 4 characters
    if len(pr) + len(pg) + len(pb) != 4:
        raise ValueError("Invalid mosaic pattern. It must contain exactly 4 characters.")

    # Assign channel indices
    num = [0] * 4
    for idx in pr:
        num[idx] = 0  # Red channel
    for idx in pg:
        num[idx] = 1  # Green channel
    for idx in pb:
        num[idx] = 2  # Blue channel

    # Step 2: Get height and width from the input image tensor
    _, _, h, w = img.size()

    # Step 3: Create an RGB mask of shape [b, h/2, w/2, 3]
    batch_size = img.size(0)
    device = img.device  # Ensure mask is on the same device as img
    mask_rgb = torch.zeros((batch_size, h // 2, w // 2, 3), device=device)

    # Fill the RGB mask according to the pattern
    mask_rgb[:, 0::2, 0::2, num[0]] = 1  # Top-left
    mask_rgb[:, 0::2, 1::2, num[1]] = 1  # Top-right
    mask_rgb[:, 1::2, 0::2, num[2]] = 1  # Bottom-left
    mask_rgb[:, 1::2, 1::2, num[3]] = 1  # Bottom-right

    # Step 4: Expand the RGB mask to a polarization mask of shape [b, h, w, 12]
    mask = torch.zeros((batch_size, h, w, 12), device=device)

    # Map RGB channels to polarization angles
    for i in range(3):  # Iterate over RGB channels
        mask[:, 1::2, 1::2, i] = mask_rgb[:, :, :, i]       # 90°
        mask[:, 0::2, 1::2, i + 3] = mask_rgb[:, :, :, i]  # 45°
        mask[:, 0::2, 0::2, i + 6] = mask_rgb[:, :, :, i]  # 135°
        mask[:, 1::2, 0::2, i + 9] = mask_rgb[:, :, :, i]  # 0°

    img = img.permute(0, 2, 3, 1)
    CPFA = mask*img

    return torch.sum(CPFA, dim=-1)


def bayer_to_rgb(bayer_img):
    """
    Convert a single-channel Bayer image to an RGB image using OpenCV.

    Args:
        bayer_img (numpy.ndarray): Input Bayer image of shape [h, w].

    Returns:
        rgb_img (numpy.ndarray): Demosaicked RGB image of shape [h, w, 3].
    """
    # Ensure the input is uint8 type
    bayer_img = (bayer_img.cpu().numpy() * 255).astype(np.uint8)

    # Apply OpenCV's demosaicking function
    rgb_img = cv2.cvtColor(bayer_img, cv2.COLOR_BAYER_RG2BGR)  # Assuming BG Bayer pattern
    rgb_img = torch.from_numpy(rgb_img.astype(np.float32) / 255.0).permute(2, 0, 1)

    return rgb_img.cuda()

def deomsaicking(img_lr,polar_angles=[90, 45, 0, 135]):
    """
    Process each polarization angle channel through demosaicing and combine them.

    Args:
        cpfa (numpy.ndarray): Input tensor of shape [b, h, w].
        polar_angles (list): List of polarization angles.

    Returns:
        result (numpy.ndarray): Tensor of shape [b, h, w, len(polar_angles)*3],
                                where each slice along the last dimension corresponds to one polarization angle's RGB channels.
    """
    batch_size = img_lr.shape[0]
    num_polar_angles = len(polar_angles)
    results = []

    for i in range(batch_size):
        resized_images = []
        for j in range(num_polar_angles):
            start_channel = j * 3
            end_channel = start_channel + 3

            rgb_img_current = img_lr[i, start_channel:end_channel].permute(1, 2, 0).cpu().numpy()

            rgb_img_current = (rgb_img_current * 255).astype(np.uint8)

            resized_img_channels = [
                cv2.resize(rgb_img_current[:, :, channel], 
                           None, 
                           fx=2, fy=2, 
                           interpolation=cv2.INTER_CUBIC)
                for channel in range(3)
            ]

            resized_img = np.stack(resized_img_channels, axis=-1)
            resized_images.append(resized_img)

        # Combine all polarized images into one tensor
        final_resized_image = np.concatenate(resized_images, axis=-1)  # Shape: [h*2, w*2, len(polar_angles)*3]
        decoded_tensor = torch.from_numpy(final_resized_image).permute(2, 0, 1).float() / 255.0
        
        results.append(decoded_tensor)

    # Stack the results from all batches into a single tensor
    final_result = torch.stack(results, dim=0)
    
    return final_result.cuda()

def deomsaicking_eit(CPFA, polar_angles=[90, 45, 0, 135]):
    """
    Process each polarization angle channel through demosaicing and combine them.

    Args:
        cpfa (numpy.ndarray): Input tensor of shape [b, h, w].
        polar_angles (list): List of polarization angles.

    Returns:
        result (numpy.ndarray): Tensor of shape [b, h, w, len(polar_angles)*3],
                                where each slice along the last dimension corresponds to one polarization angle's RGB channels.
    """
    batch_size = CPFA.shape[0]
    num_polar_angles = len(polar_angles)
    results = []

    for i in range(batch_size):
        batch_result = []

        # Split the CPFA image into separate polarization channels
        for j in range(num_polar_angles):
            # Extract one polarization channel (assuming interleaved layout)
            if j == 0:  # Top-left (0°)
                CFA = CPFA[i, 1::2, 1::2]  # Rows and columns with step 2
            elif j == 1:  # Top-right (45°)
                CFA = CPFA[i, 0::2, 1::2]
            elif j == 2:  # Bottom-left (90°)
                CFA = CPFA[i, 0::2, 0::2]
            elif j == 3:  # Bottom-right (135°)
                CFA = CPFA[i, 1::2, 0::2]

            # Perform demosaicking on this channel
            rgb_img = bayer_to_rgb(CFA)
            # rgb_full_res = torch.nn.functional.interpolate(
            #     rgb_img.unsqueeze(0), size=(height, width), mode='bilinear', align_corners=False
            # ).squeeze(0)
            batch_result.append(rgb_img)

        # Combine all polarized images into one tensor
        decoded_batch = torch.cat(batch_result, dim=0)  # Shape: [12, h, w]
        results.append(decoded_batch)

    # Stack results across batch dimension
    decoded_tensor = torch.stack(results, dim=0)  # Shape: [b, 12, h, w]

    return decoded_tensor

class init_interp(nn.Module):
    def __init__(self,
                 pattern='rggb',
                 phase='train'):
        super().__init__()
        self.pattern = pattern
        self.phase = phase

    def forward(self, img_inp: torch.Tensor):
        if self.phase in ['train']:
            CPFA = get_CPFA(img_inp, self.pattern)
            img_mr = deomsaicking_eit(CPFA)
            return img_mr
        elif self.phase in ['val','raw','GT_test']:
            img_mr = deomsaicking(img_inp)
            return img_mr