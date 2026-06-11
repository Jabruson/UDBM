import cv2
import numpy as np
import os
import glob
import argparse
import torch
from tqdm import tqdm

# ==============================================================================
# ==============================================================================

def reorder_image(img, input_order='HWC'):
    if input_order not in ['HWC', 'CHW']:
        raise ValueError(f'Wrong input_order {input_order}.')
    if input_order == 'CHW':
        img = img.transpose(1, 2, 0)
    return img

def to_y_channel(img):
    img = img.astype(np.float32) / 255.
    if img.ndim == 3 and img.shape[2] == 3:
        img = 65.481 * img[:, :, 0] + 128.553 * img[:, :, 1] + 24.966 * img[:, :, 2] + 16.0
        img = img / 255.
    return img[..., np.newaxis]

# ==============================================================================
# ==============================================================================

def calculate_psnr(img1, img2, crop_border, input_order='HWC', test_y_channel=False):
    assert img1.shape == img2.shape, (f'Image shapes differ: {img1.shape}, {img2.shape}.')
    
    if isinstance(img1, torch.Tensor):
        if img1.dim() == 4: img1 = img1.squeeze(0)
        img1 = img1.detach().cpu().numpy().transpose(1,2,0)
    if isinstance(img2, torch.Tensor):
        if img2.dim() == 4: img2 = img2.squeeze(0)
        img2 = img2.detach().cpu().numpy().transpose(1,2,0)
        
    img1 = reorder_image(img1, input_order=input_order).astype(np.float64)
    img2 = reorder_image(img2, input_order=input_order).astype(np.float64)

    if crop_border != 0:
        img1 = img1[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

    if test_y_channel:
        img1 = to_y_channel(img1)
        img2 = to_y_channel(img2)

    mse = np.mean((img1 - img2)**2)
    if mse == 0: return float('inf')
    max_value = 1. if img1.max() <= 1 else 255.
    return 20. * np.log10(max_value / np.sqrt(mse))

def _generate_3d_gaussian_kernel():
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    kernel_3 = cv2.getGaussianKernel(11, 1.5)
    kernel = torch.tensor(np.stack([window * k for k in kernel_3], axis=0))
    conv3d = torch.nn.Conv3d(1, 1, (11, 11, 11), stride=1, padding=(5, 5, 5), bias=False, padding_mode='replicate')
    conv3d.weight.requires_grad = False
    conv3d.weight[0, 0, :, :, :] = kernel
    return conv3d

def _3d_gaussian_calculator(img, conv3d):
    return conv3d(img.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)

def _ssim_3d(img1, img2, max_value):
    C1 = (0.01 * max_value) ** 2
    C2 = (0.03 * max_value) ** 2
    img1, img2 = img1.astype(np.float64), img2.astype(np.float64)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    kernel = _generate_3d_gaussian_kernel().to(device)
    t_img1 = torch.tensor(img1).float().to(device)
    t_img2 = torch.tensor(img2).float().to(device)

    mu1 = _3d_gaussian_calculator(t_img1, kernel)
    mu2 = _3d_gaussian_calculator(t_img2, kernel)
    
    mu1_sq, mu2_sq = mu1**2, mu2**2
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = _3d_gaussian_calculator(t_img1**2, kernel) - mu1_sq
    sigma2_sq = _3d_gaussian_calculator(t_img2**2, kernel) - mu2_sq
    sigma12 = _3d_gaussian_calculator(t_img1*t_img2, kernel) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return float(ssim_map.mean())

def _ssim_cly(img1, img2):
    C1, C2 = (0.01 * 255)**2, (0.03 * 255)**2
    img1, img2 = img1.astype(np.float64), img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    
    mu1 = cv2.filter2D(img1, -1, window, borderType=cv2.BORDER_REPLICATE)
    mu2 = cv2.filter2D(img2, -1, window, borderType=cv2.BORDER_REPLICATE)
    
    mu1_sq, mu2_sq = mu1**2, mu2**2
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = cv2.filter2D(img1**2, -1, window, borderType=cv2.BORDER_REPLICATE) - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window, borderType=cv2.BORDER_REPLICATE) - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window, borderType=cv2.BORDER_REPLICATE) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

def calculate_ssim(img1, img2, crop_border, input_order='HWC', test_y_channel=False):
    assert img1.shape == img2.shape, (f'Image shapes differ: {img1.shape}, {img2.shape}.')
    
    if isinstance(img1, torch.Tensor):
        if img1.dim() == 4: img1 = img1.squeeze(0)
        img1 = img1.detach().cpu().numpy().transpose(1,2,0)
    if isinstance(img2, torch.Tensor):
        if img2.dim() == 4: img2 = img2.squeeze(0)
        img2 = img2.detach().cpu().numpy().transpose(1,2,0)

    img1 = reorder_image(img1, input_order=input_order).astype(np.float64)
    img2 = reorder_image(img2, input_order=input_order).astype(np.float64)

    if crop_border != 0:
        img1 = img1[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

    if test_y_channel:
        img1 = to_y_channel(img1)
        img2 = to_y_channel(img2)
        return _ssim_cly(img1[..., 0] * 255., img2[..., 0] * 255.)

    max_value = 1 if img1.max() <= 1 else 255
    with torch.no_grad():
        final_ssim = _ssim_3d(img1, img2, max_value)
    return final_ssim

# ==============================================================================
# ==============================================================================

def get_image_paths(folder):
    """Return all image paths in a folder."""
    extensions = ['jpg', 'jpeg', 'png', 'bmp', 'tif', 'tiff']
    image_paths = []
    for ext in extensions:
        image_paths.extend(glob.glob(os.path.join(folder, f'*.{ext}')))
        image_paths.extend(glob.glob(os.path.join(folder, f'*.{ext.upper()}')))
    return sorted(image_paths)

def find_matching_file(gt_path, dist_folder):
    """
    Find a file in dist_folder with the same stem as gt_path.
    For example, 001.jpg can match 001.png or 001.bmp.
    """
    gt_filename = os.path.basename(gt_path)
    gt_stem = os.path.splitext(gt_filename)[0]
    
    direct_match = os.path.join(dist_folder, gt_filename)
    if os.path.exists(direct_match):
        return direct_match
        
    supported_exts = ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']
    for ext in supported_exts:
        candidate = os.path.join(dist_folder, gt_stem + ext)
        if os.path.exists(candidate):
            return candidate
        candidate_upper = os.path.join(dist_folder, gt_stem + ext.upper())
        if os.path.exists(candidate_upper):
            return candidate_upper
            
    return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt', type=str, default="results/RealBlur_R", help='Path to Ground Truth folder')
    parser.add_argument('--dist', type=str, default="datasets/all_in_one/Deblur/test/RealBlur_R/target", help='Path to Distorted/Restored folder')
    parser.add_argument('--crop_border', type=int, default=0, help='Crop border in pixels (e.g. 0, 4, or scale factor)')
    parser.add_argument('--y_only', default=True, help='Calculate on Y channel only (Standard for SR)')
    args = parser.parse_args()
    gt_paths = get_image_paths(args.gt)
    # PSNR: 32.8904
    # Average SSIM: 0.9272
    psnr_accum = 0.
    ssim_accum = 0.
    count = 0
    #     folder_pairs = [
    #     {
    #         'name': 'Set1', 
    #         'dist': 'results/RealBlur_R',
    #         'ref': 'datasets/all_in_one/Deblur/test/RealBlur_R/target'
    #     },
    #     {
    #         'name': 'Set2', 
    #         'dist': 'results/Reblur_J',
    #         'ref': 'datasets/all_in_one/Deblur/test/RealBlur_J/target'
    #     },
    #      {
    #         'name': 'Set3', 
    #         'dist': 'results/HIDE',
    #         'ref': 'datasets/all_in_one/Deblur/test/HIDE/target'
    #     },
    # ]
    print(f"Testing Y Channel: {args.y_only}")
    print(f"Crop Border: {args.crop_border}")
    print(f"Found {len(gt_paths)} GT images.")

    for gt_path in tqdm(gt_paths):
        dist_path = find_matching_file(gt_path, args.dist)

        if dist_path is None:
            print(f"\n[Warning] No matching file found for: {os.path.basename(gt_path)} in dist folder.")
            continue

        img_gt = cv2.imread(gt_path, cv2.IMREAD_COLOR)
        img_dist = cv2.imread(dist_path, cv2.IMREAD_COLOR)

        if img_gt is None or img_dist is None:
            print(f"\n[Error] Could not read image pair: {gt_path} or {dist_path}")
            continue

        if img_gt.shape != img_dist.shape:
            print(f"\n[Warning] Shape mismatch: {os.path.basename(gt_path)} {img_gt.shape} vs {img_dist.shape}. Skipping.")
            print(f"\n[Warning] Shape mismatch: {gt_path} {dist_path}")
            continue

        if args.y_only:
            img_gt = cv2.cvtColor(img_gt, cv2.COLOR_BGR2RGB)
            img_dist = cv2.cvtColor(img_dist, cv2.COLOR_BGR2RGB)
        
        psnr_val = calculate_psnr(img_gt, img_dist, crop_border=args.crop_border, input_order='HWC', test_y_channel=args.y_only)
        ssim_val = calculate_ssim(img_gt, img_dist, crop_border=args.crop_border, input_order='HWC', test_y_channel=args.y_only)

        psnr_accum += psnr_val
        ssim_accum += ssim_val
        count += 1

    if count > 0:
        print(f"\nResults for {count} pairs:")
        print(f"Average PSNR: {psnr_accum / count:.4f}")
        print(f"Average SSIM: {ssim_accum / count:.4f}")
    else:
        print("No valid image pairs found.")

if __name__ == '__main__':
    main()

    # 1200 32.2736 0.9240 2800 32.8904 0.9272 100h:29.2484 0.8786 100L: 33.8646 0.9540  test100: 29.4195 0.8945
    
    # 31.5393 0.9156

    # 
