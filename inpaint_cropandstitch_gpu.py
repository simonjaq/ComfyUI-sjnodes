import comfy.utils
import math
import nodes
import numpy as np
import torch
import os
import folder_paths
#import torchvision.transforms.functional as F
from PIL import Image
from scipy.ndimage import gaussian_filter, grey_dilation, binary_closing, binary_fill_holes
import torch.nn.functional as F

def rescale_i(samples, width, height, algorithm: str):
    # [B, H, W, C] → [B, C, H, W]
    samples = samples.permute(0, 3, 1, 2)

    mode = algorithm.lower()
    align_corners = False if mode in ["bilinear", "bicubic"] else None

    # Interpolate on GPU
    samples = F.interpolate(samples, size=(height, width), mode=mode, align_corners=align_corners)

    # [B, C, H, W] → [B, H, W, C]
    samples = samples.permute(0, 2, 3, 1)
    return samples


def rescale_m(samples, width, height, algorithm: str):
    if samples.ndim == 3:  # [B, H, W]
        samples = samples.unsqueeze(1)  # → [B, 1, H, W]

    mode = algorithm.lower()
    align_corners = False if mode in ["bilinear", "bicubic"] else None

    samples = F.interpolate(samples, size=(height, width), mode=mode, align_corners=align_corners)

    samples = samples.squeeze(1)  # → [B, H, W]
    return samples


def preresize_imm(image, mask, optional_context_mask, downscale_algorithm, upscale_algorithm, preresize_mode, preresize_min_width, preresize_min_height, preresize_max_width, preresize_max_height):
    current_width, current_height = image.shape[2], image.shape[1]  # Image size [batch, height, width, channels]
    
    if preresize_mode == "ensure minimum resolution":
        if current_width >= preresize_min_width and current_height >= preresize_min_height:
            return image, mask, optional_context_mask

        scale_factor_min_width = preresize_min_width / current_width
        scale_factor_min_height = preresize_min_height / current_height

        scale_factor = max(scale_factor_min_width, scale_factor_min_height)

        target_width = int(current_width * scale_factor)
        target_height = int(current_height * scale_factor)

        image = rescale_i(image, target_width, target_height, upscale_algorithm)
        mask = rescale_m(mask, target_width, target_height, 'bilinear')
        optional_context_mask = rescale_m(optional_context_mask, target_width, target_height, 'bilinear')
        
        assert target_width >= preresize_min_width and target_height >= preresize_min_height, \
            f"Internal error: After resizing, target size {target_width}x{target_height} is smaller than min size {preresize_min_width}x{preresize_min_height}"

    elif preresize_mode == "ensure minimum and maximum resolution":
        if preresize_min_width <= current_width <= preresize_max_width and preresize_min_height <= current_height <= preresize_max_height:
            return image, mask, optional_context_mask

        scale_factor_min_width = preresize_min_width / current_width
        scale_factor_min_height = preresize_min_height / current_height
        scale_factor_min = max(scale_factor_min_width, scale_factor_min_height)

        scale_factor_max_width = preresize_max_width / current_width
        scale_factor_max_height = preresize_max_height / current_height
        scale_factor_max = min(scale_factor_max_width, scale_factor_max_height)

        if scale_factor_min > 1 and scale_factor_max < 1:
            assert False, "Cannot meet both minimum and maximum resolution requirements with aspect ratio preservation."
        
        if scale_factor_min > 1:  # We're upscaling to meet min resolution
            scale_factor = scale_factor_min
            rescale_algorithm = upscale_algorithm  # Use upscale algorithm for min resolution
        else:  # We're downscaling to meet max resolution
            scale_factor = scale_factor_max
            rescale_algorithm = downscale_algorithm  # Use downscale algorithm for max resolution

        target_width = int(current_width * scale_factor)
        target_height = int(current_height * scale_factor)

        image = rescale_i(image, target_width, target_height, rescale_algorithm)
        mask = rescale_m(mask, target_width, target_height, 'nearest') # Always nearest for efficiency
        optional_context_mask = rescale_m(optional_context_mask, target_width, target_height, 'nearest') # Always nearest for efficiency
        
        assert preresize_min_width <= target_width <= preresize_max_width, \
            f"Internal error: Target width {target_width} is outside the range {preresize_min_width} - {preresize_max_width}"
        assert preresize_min_height <= target_height <= preresize_max_height, \
            f"Internal error: Target height {target_height} is outside the range {preresize_min_height} - {preresize_max_height}"

    elif preresize_mode == "ensure maximum resolution":
        if current_width <= preresize_max_width and current_height <= preresize_max_height:
            return image, mask, optional_context_mask

        scale_factor_max_width = preresize_max_width / current_width
        scale_factor_max_height = preresize_max_height / current_height
        scale_factor_max = min(scale_factor_max_width, scale_factor_max_height)

        target_width = int(current_width * scale_factor_max)
        target_height = int(current_height * scale_factor_max)

        image = rescale_i(image, target_width, target_height, downscale_algorithm)
        mask = rescale_m(mask, target_width, target_height, 'nearest')  # Always nearest for efficiency
        optional_context_mask = rescale_m(optional_context_mask, target_width, target_height, 'nearest')  # Always nearest for efficiency

        assert target_width <= preresize_max_width and target_height <= preresize_max_height, \
            f"Internal error: Target size {target_width}x{target_height} is greater than max size {preresize_max_width}x{preresize_max_height}"

    return image, mask, optional_context_mask


def fillholes_iterative_hipass_fill_m(samples):
    thresholds = [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]

    device = samples.device
    mask_np = samples.squeeze(0).detach().cpu().numpy()

    for threshold in thresholds:
        thresholded_mask = mask_np >= threshold
        closed_mask = binary_closing(thresholded_mask, structure=np.ones((3, 3)), border_value=1)
        filled_mask = binary_fill_holes(closed_mask)
        mask_np = np.maximum(mask_np, np.where(filled_mask, threshold, 0))

    final_mask = torch.from_numpy(mask_np.astype(np.float32)).unsqueeze(0).to(device)
    return final_mask


def hipassfilter_m(samples, threshold):
    return torch.where(samples >= threshold, samples, torch.zeros_like(samples))


def expand_m(mask, pixels):
    sigma = pixels / 4
    kernel_size = math.ceil(sigma * 1.5 + 1)

    # Build 2D dilation kernel
    kernel = torch.ones((1, 1, kernel_size, kernel_size), dtype=mask.dtype, device=mask.device)

    # Store original H, W
    original_shape = mask.shape[-2:]

    # Ensure input shape is [B, 1, H, W]
    if mask.ndim == 3:  # [1, H, W]
        mask = mask.unsqueeze(1)
    elif mask.ndim == 2:  # [H, W]
        mask = mask.unsqueeze(0).unsqueeze(0)
    elif mask.ndim == 4 and mask.shape[1] != 1:
        raise ValueError("expand_m expects a single-channel mask")

    # Perform dilation via convolution
    expanded = F.conv2d(mask, kernel, padding=kernel_size // 2)
    expanded = torch.clamp(expanded, 0.0, 1.0)

    # Clamp size to match original input
    expanded = expanded[..., :original_shape[0], :original_shape[1]]

    return expanded.squeeze(1)  # → [B, H, W]


def invert_m(samples):
    return 1.0 - samples


def gaussian_kernel1d(kernel_size: int, sigma: float, device) -> torch.Tensor:
    half = kernel_size // 2
    x = torch.arange(-half, half + 1, device=device).float()
    kernel = torch.exp(-0.5 * (x / sigma) ** 2)
    kernel /= kernel.sum()
    return kernel

def blur_m(mask, pixels):
    sigma = pixels / 4
    if sigma <= 0:
        return mask

    # Ensure input shape is [B, C, H, W]
    while mask.ndim > 4:
        mask = mask.squeeze(0)
    if mask.ndim == 2:
        mask = mask.unsqueeze(0).unsqueeze(0)
    elif mask.ndim == 3:
        mask = mask.unsqueeze(0)  # [1, 1, H, W]

    device = mask.device
    radius = int(3 * sigma + 0.5)
    kernel_size = 2 * radius + 1

    # Create 1D Gaussian kernel
    x = torch.arange(-radius, radius + 1, device=device).float()
    gauss = torch.exp(-0.5 * (x / sigma) ** 2)
    gauss /= gauss.sum()

    # Expand to 2D separable filters
    kernel_x = gauss.view(1, 1, 1, -1)  # shape [1,1,1,K]
    kernel_y = gauss.view(1, 1, -1, 1)  # shape [1,1,K,1]

    # Apply blur: horizontal then vertical
    padding_x = (0, radius)
    padding_y = (radius, 0)

    blurred = F.conv2d(mask, kernel_x, padding=padding_x, groups=1)
    blurred = F.conv2d(blurred, kernel_y, padding=padding_y, groups=1)

    return blurred.squeeze(1)  # shape [B, H, W]


def extend_imm(image, mask, optional_context_mask, extend_up_factor, extend_down_factor, extend_left_factor, extend_right_factor):
    B, H, W, C = image.shape

    new_H = int(H * (1.0 + extend_up_factor - 1.0 + extend_down_factor - 1.0))
    new_W = int(W * (1.0 + extend_left_factor - 1.0 + extend_right_factor - 1.0))

    assert new_H >= 0, f"Error: Trying to crop too much, height ({new_H}) must be >= 0"
    assert new_W >= 0, f"Error: Trying to crop too much, width ({new_W}) must be >= 0"

    expanded_image = torch.zeros(1, new_H, new_W, C, device=image.device)
    expanded_mask = torch.ones(1, new_H, new_W, device=mask.device)
    expanded_optional_context_mask = torch.zeros(1, new_H, new_W, device=optional_context_mask.device)

    up_padding = int(H * (extend_up_factor - 1.0))
    down_padding = new_H - H - up_padding
    left_padding = int(W * (extend_left_factor - 1.0))
    right_padding = new_W - W - left_padding

    slice_target_up = max(0, up_padding)
    slice_target_down = min(new_H, up_padding + H)
    slice_target_left = max(0, left_padding)
    slice_target_right = min(new_W, left_padding + W)

    slice_source_up = max(0, -up_padding)
    slice_source_down = min(H, new_H - up_padding)
    slice_source_left = max(0, -left_padding)
    slice_source_right = min(W, new_W - left_padding)

    image = image.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]
    expanded_image = expanded_image.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]

    expanded_image[:, :, slice_target_up:slice_target_down, slice_target_left:slice_target_right] = image[:, :, slice_source_up:slice_source_down, slice_source_left:slice_source_right]
    if up_padding > 0:
        expanded_image[:, :, :up_padding, slice_target_left:slice_target_right] = image[:, :, 0:1, slice_source_left:slice_source_right].repeat(1, 1, up_padding, 1)
    if down_padding > 0:
        expanded_image[:, :, -down_padding:, slice_target_left:slice_target_right] = image[:, :, -1:, slice_source_left:slice_source_right].repeat(1, 1, down_padding, 1)
    if left_padding > 0:
        expanded_image[:, :, :, :left_padding] = expanded_image[:, :, :, left_padding:left_padding+1].repeat(1, 1, 1, left_padding)
    if right_padding > 0:
        expanded_image[:, :, :, -right_padding:] = expanded_image[:, :, :, -right_padding-1:-right_padding].repeat(1, 1, 1, right_padding)

    expanded_mask[:, slice_target_up:slice_target_down, slice_target_left:slice_target_right] = mask[:, slice_source_up:slice_source_down, slice_source_left:slice_source_right]
    expanded_optional_context_mask[:, slice_target_up:slice_target_down, slice_target_left:slice_target_right] = optional_context_mask[:, slice_source_up:slice_source_down, slice_source_left:slice_source_right]

    expanded_image = expanded_image.permute(0, 2, 3, 1)  # [B, C, H, W] -> [B, H, W, C]
    image = image.permute(0, 2, 3, 1)  # [B, C, H, W] -> [B, H, W, C]

    return expanded_image, expanded_mask, expanded_optional_context_mask


def debug_context_location_in_image(image, x, y, w, h):
    debug_image = image.clone()
    debug_image[:, y:y+h, x:x+w, :] = 1.0 - debug_image[:, y:y+h, x:x+w, :]
    return debug_image


def findcontextarea_m(mask):
    mask_squeezed = mask[0]  # Now shape is [H, W]
    non_zero_indices = torch.nonzero(mask_squeezed)

    H, W = mask_squeezed.shape

    if non_zero_indices.numel() == 0:
        x, y = -1, -1
        w, h = -1, -1
    else:
        y, x = torch.min(non_zero_indices, dim=0).values[:2].tolist()
        y_max, x_max = torch.max(non_zero_indices, dim=0).values[:2].tolist()
        w = x_max - x + 1  # +1 to include the max index
        h = y_max - y + 1  # +1 to include the max index

    context = mask[:, y:y+h, x:x+w]
    return context, x, y, w, h


def growcontextarea_m(context, mask, x, y, w, h, extend_factor):
    img_h, img_w = mask.shape[1], mask.shape[2]

    # Compute intended growth in each direction
    grow_left = int(round(w * (extend_factor-1.0) / 2.0))
    grow_right = int(round(w * (extend_factor-1.0) / 2.0))
    grow_up = int(round(h * (extend_factor-1.0) / 2.0))
    grow_down = int(round(h * (extend_factor-1.0) / 2.0))

    # Try to grow left, but clamp at 0
    new_x = x - grow_left
    if new_x < 0:
        new_x = 0

    # Try to grow up, but clamp at 0
    new_y = y - grow_up
    if new_y < 0:
        new_y = 0

    # Right edge
    new_x2 = x + w + grow_right
    if new_x2 > img_w:
        new_x2 = img_w

    # Bottom edge
    new_y2 = y + h + grow_down
    if new_y2 > img_h:
        new_y2 = img_h

    # New width and height
    new_w = new_x2 - new_x
    new_h = new_y2 - new_y

    # Extract the context
    new_context = mask[:, new_y:new_y+new_h, new_x:new_x+new_w]

    if new_h < 0 or new_w < 0:
        new_x = 0
        new_y = 0
        new_w = mask.shape[2]
        new_h = mask.shape[1]

    return new_context, new_x, new_y, new_w, new_h


def combinecontextmask_m(context, mask, x, y, w, h, optional_context_mask):
    _, x_opt, y_opt, w_opt, h_opt = findcontextarea_m(optional_context_mask)
    if x == -1:
        x, y, w, h = x_opt, y_opt, w_opt, h_opt
    if x_opt == -1:
        x_opt, y_opt, w_opt, h_opt = x, y, w, h
    if x == -1:
        return torch.zeros(1, 0, 0, device=mask.device), -1, -1, -1, -1
    new_x = min(x, x_opt)
    new_y = min(y, y_opt)
    new_x_max = max(x + w, x_opt + w_opt)
    new_y_max = max(y + h, y_opt + h_opt)
    new_w = new_x_max - new_x
    new_h = new_y_max - new_y
    combined_context = mask[:, new_y:new_y+new_h, new_x:new_x+new_w]
    return combined_context, new_x, new_y, new_w, new_h


def pad_to_multiple(value, multiple):
    return int(math.ceil(value / multiple) * multiple)


def crop_magic_im(image, mask, x, y, w, h, target_w, target_h, padding, downscale_algorithm, upscale_algorithm):
    image = image.clone()
    mask = mask.clone()
    
    # Ok this is the most complex function in this node. The one that does the magic after all the preparation done by the other nodes.
    # Basically this function determines the right context area that encompasses the whole context area (mask+optional_context_mask),
    # that is ideally within the bounds of the original image, and that has the right aspect ratio to match target width and height.
    # It may grow the image if the aspect ratio wouldn't fit in the original image.
    # It keeps track of that growing to then be able to crop the image in the stitch node.
    # Finally, it crops the context area and resizes it to be exactly target_w and target_h.
    # It keeps track of that resize to be able to revert it in the stitch node.

    # Check for invalid inputs
    if target_w <= 0 or target_h <= 0 or w == 0 or h == 0:
        return image, 0, 0, image.shape[2], image.shape[1], image, mask, 0, 0, image.shape[2], image.shape[1]

    # Step 1: Pad target dimensions to be multiples of padding
    if padding != 0:
        target_w = pad_to_multiple(target_w, padding)
        target_h = pad_to_multiple(target_h, padding)

    # Step 2: Calculate target aspect ratio
    target_aspect_ratio = target_w / target_h

    # Step 3: Grow current context area to meet the target aspect ratio
    B, image_h, image_w, C = image.shape
    context_aspect_ratio = w / h
    if context_aspect_ratio < target_aspect_ratio:
        # Grow width to meet aspect ratio
        new_w = int(h * target_aspect_ratio)
        new_h = h
        new_x = x - (new_w - w) // 2
        new_y = y

        # Adjust new_x to keep within bounds
        if new_x < 0:
            shift = -new_x
            if new_x + new_w + shift <= image_w:
                new_x += shift
            else:
                overflow = (new_w - image_w) // 2
                new_x = -overflow
        elif new_x + new_w > image_w:
            overflow = new_x + new_w - image_w
            if new_x - overflow >= 0:
                new_x -= overflow
            else:
                overflow = (new_w - image_w) // 2
                new_x = -overflow

    else:
        # Grow height to meet aspect ratio
        new_w = w
        new_h = int(w / target_aspect_ratio)
        new_x = x
        new_y = y - (new_h - h) // 2

        # Adjust new_y to keep within bounds
        if new_y < 0:
            shift = -new_y
            if new_y + new_h + shift <= image_h:
                new_y += shift
            else:
                overflow = (new_h - image_h) // 2
                new_y = -overflow
        elif new_y + new_h > image_h:
            overflow = new_y + new_h - image_h
            if new_y - overflow >= 0:
                new_y -= overflow
            else:
                overflow = (new_h - image_h) // 2
                new_y = -overflow

    # Step 4: Grow the image to accommodate the new context area
    up_padding, down_padding, left_padding, right_padding = 0, 0, 0, 0

    expanded_image_w = image_w
    expanded_image_h = image_h

    # Adjust width for left overflow (x < 0) and right overflow (x + w > image_w)
    if new_x < 0:
        left_padding = -new_x
        expanded_image_w += left_padding
    if new_x + new_w > image_w:
        right_padding = (new_x + new_w - image_w)
        expanded_image_w += right_padding
    # Adjust height for top overflow (y < 0) and bottom overflow (y + h > image_h)
    if new_y < 0:
        up_padding = -new_y
        expanded_image_h += up_padding 
    if new_y + new_h > image_h:
        down_padding = (new_y + new_h - image_h)
        expanded_image_h += down_padding

    # Step 5: Create the new image and mask
    expanded_image = torch.zeros((image.shape[0], expanded_image_h, expanded_image_w, image.shape[3]), device=image.device)
    expanded_mask = torch.ones((mask.shape[0], expanded_image_h, expanded_image_w), device=mask.device)

    # Reorder the tensors to match the required dimension format for padding
    image = image.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]
    expanded_image = expanded_image.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]

    # Ensure the expanded image has enough room to hold the padded version of the original image
    expanded_image[:, :, up_padding:up_padding + image_h, left_padding:left_padding + image_w] = image

    # Fill the new extended areas with the edge values of the image
    if up_padding > 0:
        expanded_image[:, :, :up_padding, left_padding:left_padding + image_w] = image[:, :, 0:1, left_padding:left_padding + image_w].repeat(1, 1, up_padding, 1)
    if down_padding > 0:
        expanded_image[:, :, -down_padding:, left_padding:left_padding + image_w] = image[:, :, -1:, left_padding:left_padding + image_w].repeat(1, 1, down_padding, 1)
    if left_padding > 0:
        expanded_image[:, :, up_padding:up_padding + image_h, :left_padding] = expanded_image[:, :, up_padding:up_padding + image_h, left_padding:left_padding+1].repeat(1, 1, 1, left_padding)
    if right_padding > 0:
        expanded_image[:, :, up_padding:up_padding + image_h, -right_padding:] = expanded_image[:, :, up_padding:up_padding + image_h, -right_padding-1:-right_padding].repeat(1, 1, 1, right_padding)

    # Reorder the tensors back to [B, H, W, C] format
    expanded_image = expanded_image.permute(0, 2, 3, 1)  # [B, C, H, W] -> [B, H, W, C]
    image = image.permute(0, 2, 3, 1)  # [B, C, H, W] -> [B, H, W, C]

    # Same for the mask
    expanded_mask[:, up_padding:up_padding + image_h, left_padding:left_padding + image_w] = mask

    # Record the cto values (canvas to original)
    cto_x = left_padding
    cto_y = up_padding
    cto_w = image_w
    cto_h = image_h

    # The final expanded image and mask
    canvas_image = expanded_image
    canvas_mask = expanded_mask

    # Step 6: Crop the image and mask around x, y, w, h
    ctc_x = new_x+left_padding
    ctc_y = new_y+up_padding
    ctc_w = new_w
    ctc_h = new_h

    # Crop the image and mask
    cropped_image = canvas_image[:, ctc_y:ctc_y + ctc_h, ctc_x:ctc_x + ctc_w]
    cropped_mask = canvas_mask[:, ctc_y:ctc_y + ctc_h, ctc_x:ctc_x + ctc_w]

    # Step 7: Resize image and mask to the target width and height
    # Decide which algorithm to use based on the scaling direction
    if target_w > ctc_w or target_h > ctc_h:  # Upscaling
        cropped_image = rescale_i(cropped_image, target_w, target_h, upscale_algorithm)
        cropped_mask = rescale_m(cropped_mask, target_w, target_h, upscale_algorithm)
    else:  # Downscaling
        cropped_image = rescale_i(cropped_image, target_w, target_h, downscale_algorithm)
        cropped_mask = rescale_m(cropped_mask, target_w, target_h, downscale_algorithm)

    return canvas_image, cto_x, cto_y, cto_w, cto_h, cropped_image, cropped_mask, ctc_x, ctc_y, ctc_w, ctc_h


def stitch_magic_im(canvas_image, inpainted_image, mask, ctc_x, ctc_y, ctc_w, ctc_h, cto_x, cto_y, cto_w, cto_h, downscale_algorithm, upscale_algorithm):
    canvas_image = canvas_image.clone()
    inpainted_image = inpainted_image.clone()
    mask = mask.clone()

    # Resize inpainted image and mask to match the context size
    _, h, w, _ = inpainted_image.shape
    if ctc_w > w or ctc_h > h:  # Upscaling
        resized_image = rescale_i(inpainted_image, ctc_w, ctc_h, upscale_algorithm)
        resized_mask = rescale_m(mask, ctc_w, ctc_h, upscale_algorithm)
    else:  # Downscaling
        resized_image = rescale_i(inpainted_image, ctc_w, ctc_h, downscale_algorithm)
        resized_mask = rescale_m(mask, ctc_w, ctc_h, downscale_algorithm)

    # Clamp mask to [0, 1] and expand to match image channels
    resized_mask = resized_mask.clamp(0, 1).unsqueeze(-1)  # shape: [1, H, W, 1]

    #print("resized_mask stats:", resized_mask.min().item(), resized_mask.max().item(), resized_mask.mean().item())

    # Extract the canvas region we're about to overwrite
    canvas_crop = canvas_image[:, ctc_y:ctc_y + ctc_h, ctc_x:ctc_x + ctc_w]

    # Blend: new = mask * inpainted + (1 - mask) * canvas
    blended = resized_mask * resized_image + (1.0 - resized_mask) * canvas_crop

    # Paste the blended region back onto the canvas
    canvas_image[:, ctc_y:ctc_y + ctc_h, ctc_x:ctc_x + ctc_w] = blended

    # Final crop to get back the original image area
    output_image = canvas_image[:, cto_y:cto_y + cto_h, cto_x:cto_x + cto_w]

    return output_image


class InpaintCropImproved:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # Required inputs
                "image": ("IMAGE",),

                # Resize algorithms
                "downscale_algorithm": (["nearest", "bilinear", "bicubic", "area", "nearest-exact"], {"default": "bilinear"}),
                "upscale_algorithm":   (["nearest", "bilinear", "bicubic", "area", "nearest-exact"], {"default": "bicubic"}),

                # Pre-resize input image
                "preresize": ("BOOLEAN", {"default": False, "tooltip": "Resize the original image before processing."}),
                "preresize_mode": (["ensure minimum resolution", "ensure maximum resolution", "ensure minimum and maximum resolution"], {"default": "ensure minimum resolution"}),
                "preresize_min_width": ("INT", {"default": 1024, "min": 0, "max": nodes.MAX_RESOLUTION, "step": 1}),
                "preresize_min_height": ("INT", {"default": 1024, "min": 0, "max": nodes.MAX_RESOLUTION, "step": 1}),
                "preresize_max_width": ("INT", {"default": nodes.MAX_RESOLUTION, "min": 0, "max": nodes.MAX_RESOLUTION, "step": 1}),
                "preresize_max_height": ("INT", {"default": nodes.MAX_RESOLUTION, "min": 0, "max": nodes.MAX_RESOLUTION, "step": 1}),

                # Mask manipulation
                "mask_fill_holes": ("BOOLEAN", {"default": True, "tooltip": "Mark as masked any areas fully enclosed by mask."}),
                "mask_expand_pixels": ("INT", {"default": 0, "min": 0, "max": nodes.MAX_RESOLUTION, "step": 1, "tooltip": "Expand the mask by a certain amount of pixels before processing."}),
                "mask_invert": ("BOOLEAN", {"default": False,"tooltip": "Invert mask so that anything masked will be kept."}),
                "mask_blend_pixels": ("INT", {"default": 32, "min": 0, "max": 64, "step": 1, "tooltip": "How many pixels to blend into the original image."}),
                "mask_hipass_filter": ("FLOAT", {"default": 0.1, "min": 0, "max": 1, "step": 0.01, "tooltip": "Ignore mask values lower than this value."}),

                # Extend image for outpainting
                "extend_for_outpainting": ("BOOLEAN", {"default": False, "tooltip": "Extend the image for outpainting."}),
                "extend_up_factor": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 100.0, "step": 0.01}),
                "extend_down_factor": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 100.0, "step": 0.01}),
                "extend_left_factor": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 100.0, "step": 0.01}),
                "extend_right_factor": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 100.0, "step": 0.01}),

                # Context
                "context_from_mask_extend_factor": ("FLOAT", {"default": 1.2, "min": 1.0, "max": 100.0, "step": 0.01, "tooltip": "Grow the context area from the mask by a certain factor in every direction. For example, 1.5 grabs extra 50% up, down, left, and right as context."}),

                # Output
                "output_resize_to_target_size": ("BOOLEAN", {"default": True, "tooltip": "Force a specific resolution for sampling."}),
                "output_target_width": ("INT", {"default": 512, "min": 64, "max": nodes.MAX_RESOLUTION, "step": 1}),
                "output_target_height": ("INT", {"default": 512, "min": 64, "max": nodes.MAX_RESOLUTION, "step": 1}),
                "output_padding": (["0", "8", "16", "32", "64", "128", "256", "512"], {"default": "32"}),
           },
           "optional": {
                # Optional inputs
                "mask": ("MASK",),
                "optional_context_mask": ("MASK",),
           }
        }

    FUNCTION = "inpaint_crop"
    CATEGORY = "sjnodes"
    DESCRIPTION = "Crops an image around a mask for inpainting, the optional context mask defines an extra area to keep for the context."


    # Remove the following # to turn on debug mode (extra outputs, print statements)
    #'''
    DEBUG_MODE = False
    RETURN_TYPES = ("STITCHER", "IMAGE", "MASK")
    RETURN_NAMES = ("stitcher", "cropped_image", "cropped_mask")

    '''
    
    DEBUG_MODE = True # TODO
    RETURN_TYPES = ("STITCHER", "IMAGE", "MASK",
        # DEBUG
        "IMAGE",
        "MASK",
        "MASK",
        "MASK",
        "MASK",
        "MASK",
        "MASK",
        "IMAGE",
        "MASK",
        "MASK",
        "IMAGE",
        "MASK",
        "IMAGE",
        "MASK",
        "IMAGE",
        "MASK",
        "IMAGE",
        "IMAGE",
        "MASK",
        "IMAGE",
        "IMAGE",
        "IMAGE",
        "MASK",
    )
    RETURN_NAMES = ("stitcher", "cropped_image", "cropped_mask",
        # DEBUG
        "DEBUG_preresize_image",
        "DEBUG_preresize_mask",
        "DEBUG_fillholes_mask",
        "DEBUG_expand_mask",
        "DEBUG_invert_mask",
        "DEBUG_blur_mask",
        "DEBUG_hipassfilter_mask",
        "DEBUG_extend_image",
        "DEBUG_extend_mask",
        "DEBUG_context_from_mask",
        "DEBUG_context_from_mask_location",
        "DEBUG_context_expand",
        "DEBUG_context_expand_location",
        "DEBUG_context_with_context_mask",
        "DEBUG_context_with_context_mask_location",
        "DEBUG_context_to_target",
        "DEBUG_context_to_target_location",
        "DEBUG_context_to_target_image",
        "DEBUG_context_to_target_mask",
        "DEBUG_canvas_image",
        "DEBUG_orig_in_canvas_location",
        "DEBUG_cropped_in_canvas_location",
        "DEBUG_cropped_mask_blend",
    )
    #'''

 
    def inpaint_crop(self, image, downscale_algorithm, upscale_algorithm, preresize, preresize_mode, preresize_min_width, preresize_min_height, preresize_max_width, preresize_max_height, extend_for_outpainting, extend_up_factor, extend_down_factor, extend_left_factor, extend_right_factor, mask_hipass_filter, mask_fill_holes, mask_expand_pixels, mask_invert, mask_blend_pixels, context_from_mask_extend_factor, output_resize_to_target_size, output_target_width, output_target_height, output_padding, mask=None, optional_context_mask=None):
        image = image.clone()
        if mask is not None:
            mask = mask.clone()
        if optional_context_mask is not None:
            optional_context_mask = optional_context_mask.clone()

        output_padding = int(output_padding)
        
        # Check that some parameters make sense
        if preresize and preresize_mode == "ensure minimum and maximum resolution":
            assert preresize_max_width >= preresize_min_width, "Preresize maximum width must be greater than or equal to minimum width"
            assert preresize_max_height >= preresize_min_height, "Preresize maximum height must be greater than or equal to minimum height"

        if self.DEBUG_MODE:
            print('Inpaint Crop Batch input')
            print(image.shape, type(image), image.dtype)
            if mask is not None:
                print(mask.shape, type(mask), mask.dtype)
            if optional_context_mask is not None:
                print(optional_context_mask.shape, type(optional_context_mask), optional_context_mask.dtype)

        if image.shape[0] > 1:
            assert output_resize_to_target_size, "output_resize_to_target_size must be enabled when input is a batch of images, given all images in the batch output have to be the same size"

        # When a LoadImage node passes a mask without user editing, it may be the wrong shape.
        # Detect and fix that to avoid shape mismatch errors.
        if mask is not None and (image.shape[0] == 1 or mask.shape[0] == 1 or mask.shape[0] == image.shape[0]):
            if mask.shape[1] != image.shape[1] or mask.shape[2] != image.shape[2]:
                if torch.count_nonzero(mask) == 0:
                    mask = torch.zeros((mask.shape[0], image.shape[1], image.shape[2]), device=image.device, dtype=image.dtype)

        if optional_context_mask is not None and (image.shape[0] == 1 or optional_context_mask.shape[0] == 1 or optional_context_mask.shape[0] == image.shape[0]):
            if optional_context_mask.shape[1] != image.shape[1] or optional_context_mask.shape[2] != image.shape[2]:
                if torch.count_nonzero(optional_context_mask) == 0:
                    optional_context_mask = torch.zeros((optional_context_mask.shape[0], image.shape[1], image.shape[2]), device=image.device, dtype=image.dtype)

        # If no mask is provided, create one with the shape of the image
        if mask is None:
            mask = torch.zeros_like(image[:, :, :, 0])
    
        # If there is only one image for many masks, replicate it for all masks
        if mask.shape[0] > 1 and image.shape[0] == 1:
            assert image.dim() == 4, f"Expected 4D BHWC image tensor, got {image.shape}"
            image = image.expand(mask.shape[0], -1, -1, -1).clone()

        # If there is only one mask for many images, replicate it for all images
        if image.shape[0] > 1 and mask.shape[0] == 1:
            assert mask.dim() == 3, f"Expected 3D BHW mask tensor, got {mask.shape}"
            mask = mask.expand(image.shape[0], -1, -1).clone()

        # If no optional_context_mask is provided, create one with the shape of the image
        if optional_context_mask is None:
            optional_context_mask = torch.zeros_like(image[:, :, :, 0])

        # If there is only one optional_context_mask for many images, replicate it for all images
        if image.shape[0] > 1 and optional_context_mask.shape[0] == 1:
            assert optional_context_mask.dim() == 3, f"Expected 3D BHW optional_context_mask tensor, got {optional_context_mask.shape}"
            optional_context_mask = optional_context_mask.expand(image.shape[0], -1, -1).clone()

        if self.DEBUG_MODE:
            print('Inpaint Crop Batch ready')
            print(image.shape, type(image), image.dtype)
            print(mask.shape, type(mask), mask.dtype)
            print(optional_context_mask.shape, type(optional_context_mask), optional_context_mask.dtype)

         # Validate data
        assert image.ndimension() == 4, f"Expected 4 dimensions for image, got {image.ndimension()}"
        assert mask.ndimension() == 3, f"Expected 3 dimensions for mask, got {mask.ndimension()}"
        assert optional_context_mask.ndimension() == 3, f"Expected 3 dimensions for optional_context_mask, got {optional_context_mask.ndimension()}"
        assert mask.shape[1:] == image.shape[1:3], f"Mask dimensions do not match image dimensions. Expected {image.shape[1:3]}, got {mask.shape[1:]}"
        assert optional_context_mask.shape[1:] == image.shape[1:3], f"optional_context_mask dimensions do not match image dimensions. Expected {image.shape[1:3]}, got {optional_context_mask.shape[1:]}"
        assert mask.shape[0] == image.shape[0], f"Mask batch does not match image batch. Expected {image.shape[0]}, got {mask.shape[0]}"
        assert optional_context_mask.shape[0] == image.shape[0], f"Optional context mask batch does not match image batch. Expected {image.shape[0]}, got {optional_context_mask.shape[0]}"

        # Run for each image separately
        result_stitcher = {
            'downscale_algorithm': downscale_algorithm,
            'upscale_algorithm': upscale_algorithm,
            'blend_pixels': mask_blend_pixels,
            'canvas_to_orig_x': [],
            'canvas_to_orig_y': [],
            'canvas_to_orig_w': [],
            'canvas_to_orig_h': [],
            'canvas_image': [],
            'cropped_to_canvas_x': [],
            'cropped_to_canvas_y': [],
            'cropped_to_canvas_w': [],
            'cropped_to_canvas_h': [],
            'cropped_mask_for_blend': [],
        }
        
        result_image = []
        result_mask = []

        debug_outputs = {name: [] for name in self.RETURN_NAMES if name.startswith("DEBUG_")}

        batch_size = image.shape[0]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        image_dtype = torch.float32 if torch.cuda.is_available() else torch.float32
        mask_dtype = torch.float32 if torch.cuda.is_available() else torch.float32

        for b in range(batch_size):
            one_image = image[b].unsqueeze(0).to(device=device, dtype=image_dtype)
            one_mask = mask[b].unsqueeze(0).to(device=device, dtype=mask_dtype)
            one_optional_context_mask = optional_context_mask[b].unsqueeze(0).to(device=device, dtype=mask_dtype)

            outputs = self.inpaint_crop_single_image(
                one_image, downscale_algorithm, upscale_algorithm, preresize, preresize_mode,
                preresize_min_width, preresize_min_height, preresize_max_width, preresize_max_height,
                extend_for_outpainting, extend_up_factor, extend_down_factor, extend_left_factor, extend_right_factor,
                mask_hipass_filter, mask_fill_holes, mask_expand_pixels, mask_invert, mask_blend_pixels,
                context_from_mask_extend_factor, output_resize_to_target_size, output_target_width, output_target_height,
                output_padding, one_mask, one_optional_context_mask)

            stitcher, cropped_image, cropped_mask = outputs[:3]

            for key in [
                'canvas_to_orig_x', 'canvas_to_orig_y', 'canvas_to_orig_w', 'canvas_to_orig_h',
                'canvas_image', 'cropped_to_canvas_x', 'cropped_to_canvas_y',
                'cropped_to_canvas_w', 'cropped_to_canvas_h', 'cropped_mask_for_blend'
            ]:
                value = stitcher[key]
                if isinstance(value, torch.Tensor):
                    value = value.cpu()

                    # Optimize memory for large tensors
                    if key == 'canvas_image':
                        value = value.to(torch.float16)  # still normalized float16
                    elif key == 'cropped_mask_for_blend':
                        value = (value.clamp(0, 1) * 255).to(torch.uint8)  # convert to uint8 mask
                    else:
                        value = value.clone()

                result_stitcher[key].append(value)

            result_image.append(cropped_image.cpu().squeeze(0).to(torch.float16))
            result_mask.append((cropped_mask.cpu().squeeze(0).clamp(0, 1) * 255).to(torch.uint8))

            for name, output in zip(self.RETURN_NAMES[3:], outputs[3:]):
                if name.startswith("DEBUG_"):
                    debug_outputs[name].append(output.cpu().squeeze(0))

        result_image = torch.stack(result_image, dim=0)
        result_mask = torch.stack(result_mask, dim=0)

        if self.DEBUG_MODE:
            print('Inpaint Crop Batch output')
            print(result_image.shape, type(result_image), result_image.dtype)
            print(result_mask.shape, type(result_mask), result_mask.dtype)

        debug_outputs = {name: torch.stack(values, dim=0) for name, values in debug_outputs.items()}

        # Free up memory
        del one_image, one_mask, one_optional_context_mask, outputs, stitcher, value, cropped_image, cropped_mask
        torch.cuda.empty_cache()

        return result_stitcher, result_image, result_mask, *[debug_outputs[name] for name in self.RETURN_NAMES if name.startswith("DEBUG_")]


    def inpaint_crop_single_image(self, image, downscale_algorithm, upscale_algorithm, preresize, preresize_mode, preresize_min_width, preresize_min_height, preresize_max_width, preresize_max_height, extend_for_outpainting, extend_up_factor, extend_down_factor, extend_left_factor, extend_right_factor, mask_hipass_filter, mask_fill_holes, mask_expand_pixels, mask_invert, mask_blend_pixels, context_from_mask_extend_factor, output_resize_to_target_size, output_target_width, output_target_height, output_padding, mask, optional_context_mask):
        if preresize:
            image, mask, optional_context_mask = preresize_imm(image, mask, optional_context_mask, downscale_algorithm, upscale_algorithm, preresize_mode, preresize_min_width, preresize_min_height, preresize_max_width, preresize_max_height)
        if self.DEBUG_MODE:
            DEBUG_preresize_image = image.clone()
            DEBUG_preresize_mask = mask.clone()
       
        if mask_fill_holes:
           mask = fillholes_iterative_hipass_fill_m(mask)
        if self.DEBUG_MODE:
            DEBUG_fillholes_mask = mask.clone()

        if mask_expand_pixels > 0:
            mask = expand_m(mask, mask_expand_pixels)
        if self.DEBUG_MODE:
            DEBUG_expand_mask = mask.clone()

        if mask_invert:
            mask = invert_m(mask)
        if self.DEBUG_MODE:
            DEBUG_invert_mask = mask.clone()

        if mask_blend_pixels > 0:
            mask = expand_m(mask, mask_blend_pixels)
            mask = blur_m(mask, mask_blend_pixels*0.5)
        if self.DEBUG_MODE:
            DEBUG_blur_mask = mask.clone()

        if mask_hipass_filter >= 0.01:
            mask = hipassfilter_m(mask, mask_hipass_filter)
            optional_context_mask = hipassfilter_m(optional_context_mask, mask_hipass_filter)
        if self.DEBUG_MODE:
            DEBUG_hipassfilter_mask = mask.clone()

        if extend_for_outpainting:
            image, mask, optional_context_mask = extend_imm(image, mask, optional_context_mask, extend_up_factor, extend_down_factor, extend_left_factor, extend_right_factor)
        if self.DEBUG_MODE:
            DEBUG_extend_image = image.clone()
            DEBUG_extend_mask = mask.clone()

        context, x, y, w, h = findcontextarea_m(mask)
        # If no mask, mask everything for some inpainting.
        if x == -1 or w == -1 or h == -1 or y == -1:
            x, y, w, h = 0, 0, image.shape[2], image.shape[1]
            context = mask[:, y:y+h, x:x+w]
        if self.DEBUG_MODE:
            DEBUG_context_from_mask = context.clone()
            DEBUG_context_from_mask_location = debug_context_location_in_image(image, x, y, w, h)

        if context_from_mask_extend_factor >= 1.01:
            context, x, y, w, h = growcontextarea_m(context, mask, x, y, w, h, context_from_mask_extend_factor)
        # If no mask, mask everything for some inpainting.
        if x == -1 or w == -1 or h == -1 or y == -1:
            x, y, w, h = 0, 0, image.shape[2], image.shape[1]
            context = mask[:, y:y+h, x:x+w]
        if self.DEBUG_MODE:
            DEBUG_context_expand = context.clone()
            DEBUG_context_expand_location = debug_context_location_in_image(image, x, y, w, h)

        context, x, y, w, h = combinecontextmask_m(context, mask, x, y, w, h, optional_context_mask)
        # If no mask, mask everything for some inpainting.
        if x == -1 or w == -1 or h == -1 or y == -1:
            x, y, w, h = 0, 0, image.shape[2], image.shape[1]
            context = mask[:, y:y+h, x:x+w]
        if self.DEBUG_MODE:
            DEBUG_context_with_context_mask = context.clone()
            DEBUG_context_with_context_mask_location = debug_context_location_in_image(image, x, y, w, h)

        if not output_resize_to_target_size:
            canvas_image, cto_x, cto_y, cto_w, cto_h, cropped_image, cropped_mask, ctc_x, ctc_y, ctc_w, ctc_h = crop_magic_im(image, mask, x, y, w, h, w, h, output_padding, downscale_algorithm, upscale_algorithm)
        else: # if output_resize_to_target_size:
            canvas_image, cto_x, cto_y, cto_w, cto_h, cropped_image, cropped_mask, ctc_x, ctc_y, ctc_w, ctc_h = crop_magic_im(image, mask, x, y, w, h, output_target_width, output_target_height, output_padding, downscale_algorithm, upscale_algorithm)
        if self.DEBUG_MODE:
            DEBUG_context_to_target = context.clone()
            DEBUG_context_to_target_location = debug_context_location_in_image(image, x, y, w, h)
            DEBUG_context_to_target_image = image.clone()
            DEBUG_context_to_target_mask = mask.clone()
            DEBUG_canvas_image = canvas_image.clone()
            DEBUG_orig_in_canvas_location = debug_context_location_in_image(canvas_image, cto_x, cto_y, cto_w, cto_h)
            DEBUG_cropped_in_canvas_location = debug_context_location_in_image(canvas_image, ctc_x, ctc_y, ctc_w, ctc_h)

        # For blending, grow the mask even further and make it blurrier.
        cropped_mask_blend = cropped_mask.clone()
        if mask_blend_pixels > 0:
           cropped_mask_blend = blur_m(cropped_mask_blend, mask_blend_pixels*0.5)
        if self.DEBUG_MODE:
            DEBUG_cropped_mask_blend = cropped_mask_blend.clone()

        stitcher = {
            'canvas_to_orig_x': cto_x,
            'canvas_to_orig_y': cto_y,
            'canvas_to_orig_w': cto_w,
            'canvas_to_orig_h': cto_h,
            'canvas_image': canvas_image,
            'cropped_to_canvas_x': ctc_x,
            'cropped_to_canvas_y': ctc_y,
            'cropped_to_canvas_w': ctc_w,
            'cropped_to_canvas_h': ctc_h,
            'cropped_mask_for_blend': cropped_mask_blend,
        }

        if not self.DEBUG_MODE:
            return stitcher, cropped_image, cropped_mask
        else:
            return stitcher, cropped_image, cropped_mask, DEBUG_preresize_image, DEBUG_preresize_mask, DEBUG_fillholes_mask, DEBUG_expand_mask, DEBUG_invert_mask, DEBUG_blur_mask, DEBUG_hipassfilter_mask, DEBUG_extend_image, DEBUG_extend_mask, DEBUG_context_from_mask, DEBUG_context_from_mask_location, DEBUG_context_expand, DEBUG_context_expand_location, DEBUG_context_with_context_mask, DEBUG_context_with_context_mask_location, DEBUG_context_to_target, DEBUG_context_to_target_location, DEBUG_context_to_target_image, DEBUG_context_to_target_mask, DEBUG_canvas_image, DEBUG_orig_in_canvas_location, DEBUG_cropped_in_canvas_location, DEBUG_cropped_mask_blend




class InpaintStitchImproved:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "stitcher": ("STITCHER",),
                "inpainted_image": ("IMAGE",),
            }
        }

    CATEGORY = "sjnodes"
    DESCRIPTION = "Stitches an image cropped with Inpaint Crop back into the original image"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "inpaint_stitch"

    def inpaint_stitch(self, stitcher, inpainted_image):
        inpainted_image = inpainted_image.clone()
        results = []

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        inpainted_image = inpainted_image.to(device)

        batch_size = inpainted_image.shape[0]
        override = len(stitcher['cropped_to_canvas_x']) == 1

        for b in range(batch_size):
            one_image = inpainted_image[b].unsqueeze(0)
            one_stitcher = {}

            # Copy static (non-batched) stitch parameters
            for key in ['downscale_algorithm', 'upscale_algorithm', 'blend_pixels']:
                one_stitcher[key] = stitcher[key]

            # Copy and convert per-sample fields
            for key in [
                'canvas_to_orig_x', 'canvas_to_orig_y', 'canvas_to_orig_w', 'canvas_to_orig_h',
                'canvas_image', 'cropped_to_canvas_x', 'cropped_to_canvas_y',
                'cropped_to_canvas_w', 'cropped_to_canvas_h', 'cropped_mask_for_blend'
            ]:
                value = stitcher[key][0] if override else stitcher[key][b]

                if isinstance(value, torch.Tensor):
                    value = value.to(device)
                    if key == 'canvas_image' and value.dtype == torch.float16:
                        value = value.to(dtype=torch.float32)
                    elif key == 'cropped_mask_for_blend' and value.dtype == torch.uint8:
                        value = value.to(dtype=torch.float32) / 255.0

                one_stitcher[key] = value

            stitched_image, = self.inpaint_stitch_single_image(one_stitcher, one_image)
            results.append(stitched_image.squeeze(0).clone().cpu())

        result_batch = torch.stack(results, dim=0)
        return (result_batch,)

    def inpaint_stitch_single_image(self, stitcher, inpainted_image):
        return (stitch_magic_im(
            stitcher['canvas_image'], inpainted_image, stitcher['cropped_mask_for_blend'],
            stitcher['cropped_to_canvas_x'], stitcher['cropped_to_canvas_y'],
            stitcher['cropped_to_canvas_w'], stitcher['cropped_to_canvas_h'],
            stitcher['canvas_to_orig_x'], stitcher['canvas_to_orig_y'],
            stitcher['canvas_to_orig_w'], stitcher['canvas_to_orig_h'],
            stitcher['downscale_algorithm'], stitcher['upscale_algorithm']
        ),)
    



class SaveStitcherToFile:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "stitcher": ("STITCHER",),
                "filename": ("STRING", {"default": "stitcher_backup.pt"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("file_path",)
    FUNCTION = "save"
    CATEGORY = "sjnodes/utils"

    def save(self, stitcher, filename):
        output_dir = folder_paths.get_output_directory()
        os.makedirs(output_dir, exist_ok=True)

        full_path = os.path.join(output_dir, filename)

        safe_stitcher = {}
        for k, v in stitcher.items():
            if isinstance(v, list):
                safe_stitcher[k] = [x.cpu() if isinstance(x, torch.Tensor) else x for x in v]
            else:
                safe_stitcher[k] = v

        torch.save(safe_stitcher, full_path)

        return (full_path,)


class LoadStitcherFromFile:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "file_path": ("STRING", {"default": os.path.join("output", "stitcher_backup.pt")}),
            }
        }

    RETURN_TYPES = ("STITCHER",)
    RETURN_NAMES = ("stitcher",)
    FUNCTION = "load"
    CATEGORY = "sjnodes/utils"

    def load(self, file_path):
        stitcher = torch.load(file_path, map_location="cpu")
        return (stitcher,)
    


class SmoothTemporalMask:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "masks": ("MASK",),
                "mode": (["mask", "box"], {"default": "mask"}),
                "smoothing_window": ("INT", {"default": 5, "min": 1, "max": 99}),
                "threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0}),
                "min_expansion_px": ("INT", {"default": 4, "min": 0, "max": 64}),
                "smoothing_alpha": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0}),
            }
        }

    CATEGORY = "sjnodes/mask"
    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("smoothed_mask",)
    FUNCTION = "smooth"

    def smooth(self, masks, mode, smoothing_window, threshold, min_expansion_px, smoothing_alpha):
        if masks.dim() == 4 and masks.shape[1] == 1:
            masks = masks[:, 0]
        elif masks.dim() != 3:
            raise ValueError("Expected mask tensor of shape [B, H, W] or [B, 1, H, W]")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        masks = masks.to(device)
        B, H, W = masks.shape
        dtype = masks.dtype

        if mode == "mask":
            if min_expansion_px > 0:
                kernel_size = min_expansion_px * 2 + 1
                kernel = torch.ones((1, 1, kernel_size, kernel_size), device=device, dtype=dtype)
                expanded = F.conv2d(masks.unsqueeze(1), kernel, padding=min_expansion_px)
                expanded = (expanded > 0).float().squeeze(1)
            else:
                expanded = masks

            temporal = expanded.permute(1, 2, 0).reshape(-1, 1, B)
            kernel = torch.ones(1, 1, smoothing_window, device=device, dtype=dtype) / smoothing_window
            smoothed = F.conv1d(temporal, kernel, padding=smoothing_window // 2)
            smoothed = smoothed.reshape(H, W, B).permute(2, 0, 1)
            result = (smoothed > threshold).float().to("cpu")

        elif mode == "box":
            result = torch.zeros_like(masks, device="cpu")
            prev_cx, prev_cy, prev_w, prev_h = None, None, None, None

            for i in range(B):
                mask = masks[i]
                nonzero = torch.nonzero(mask > threshold, as_tuple=False)
                if nonzero.numel() == 0:
                    continue

                y_min, x_min = nonzero.min(dim=0).values
                y_max, x_max = nonzero.max(dim=0).values

                cx = (x_min + x_max) / 2.0
                cy = (y_min + y_max) / 2.0
                w = x_max - x_min
                h = y_max - y_min

                if prev_cx is None:
                    scx, scy, sw, sh = cx, cy, w, h
                else:
                    scx = smoothing_alpha * cx + (1 - smoothing_alpha) * prev_cx
                    scy = smoothing_alpha * cy + (1 - smoothing_alpha) * prev_cy
                    sw = smoothing_alpha * w + (1 - smoothing_alpha) * prev_w
                    sh = smoothing_alpha * h + (1 - smoothing_alpha) * prev_h

                prev_cx, prev_cy, prev_w, prev_h = scx, scy, sw, sh

                half_w = sw / 2 + min_expansion_px
                half_h = sh / 2 + min_expansion_px
                x1 = int(torch.clamp(scx - half_w, 0, W - 1))
                x2 = int(torch.clamp(scx + half_w, 0, W - 1))
                y1 = int(torch.clamp(scy - half_h, 0, H - 1))
                y2 = int(torch.clamp(scy + half_h, 0, H - 1))

                result[i, y1:y2, x1:x2] = 1.0

        torch.cuda.empty_cache()
        return (result,)
    

class CrossFadeVideo:
    """
    Crossfades two batches of images over a specified number of frames.
    The output sequence contains:
    - all frames from images_1 except the last `fade_len`
    - `fade_len` blended frames between images_1 and images_2
    - all frames from images_2 starting after the `fade_len`
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images_1": ("IMAGE",),
                "images_2": ("IMAGE",),
                "fade_len": ("INT", {"default": 8, "min": 1, "max": 1024}),
                "interpolation": (["linear", "ease_in", "ease_out", "ease_in_out"],),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("faded_video",)
    FUNCTION = "crossfade_video"
    CATEGORY = "video/transition"

    def crossfade_video(self, images_1, images_2, fade_len, interpolation):
        assert images_1.shape[0] >= fade_len, "images_1 too short"
        assert images_2.shape[0] >= fade_len, "images_2 too short"

        start_1 = images_1[:-fade_len]
        end_1 = images_1[-fade_len:]
        start_2 = images_2[:fade_len]
        rest_2 = images_2[fade_len:]

        easing_fn = self.get_easing_fn(interpolation)
        alphas = torch.linspace(0, 1, fade_len)
        alphas = torch.tensor([easing_fn(a.item()) for a in alphas], dtype=images_1.dtype)

        fade = [(1 - alpha) * img1 + alpha * img2 for img1, img2, alpha in zip(end_1, start_2, alphas)]
        fade = torch.stack(fade)

        result = torch.cat([start_1, fade, rest_2], dim=0)
        return (result,)

    def get_easing_fn(self, name):
        if name == "linear":
            return lambda t: t
        elif name == "ease_in":
            return lambda t: t * t
        elif name == "ease_out":
            return lambda t: 1 - (1 - t) * (1 - t)
        elif name == "ease_in_out":
            return lambda t: 3 * t * t - 2 * t * t * t
        else:
            return lambda t: t  # fallback
