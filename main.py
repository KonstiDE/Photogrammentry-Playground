import utils, sift
from jit_ransac import ransac, ratio_test_threshold_match
from torchvision.io import read_image, ImageReadMode
import torchvision.transforms.v2 as T
from PIL import Image
import torch
from tqdm.auto import tqdm
from tensordict import tensorclass, MemoryMappedTensor
import matplotlib.pyplot as plt

import kornia
import kornia.geometry.transform as ktf
import torchvision.transforms.functional as F
import torch.nn.functional as Fnn


@tensorclass
class KeyPoints:
    pts: torch.Tensor
    desc: torch.Tensor

    @classmethod
    def from_images(cls, images: torch.Tensor, num_features=512, patch_size=41, angle_bins=8, spatial_bins=8,
                    batch_size: int = 8, device: str = 'cpu'):
        descs = MemoryMappedTensor.zeros(images.size(0), num_features, 8 * angle_bins * spatial_bins,
                                         dtype=torch.float32)
        kpts = MemoryMappedTensor.zeros(images.size(0), num_features, 2, dtype=torch.float32)

        model = sift.SIFT(num_features=num_features, patch_size=patch_size, angle_bins=angle_bins,
                          spatial_bins=spatial_bins)
        model.to(device)
        with torch.no_grad():
            for i in tqdm(range(0, images.size(0), batch_size)):
                batch = images[i:i + batch_size].to(device)
                k, d = model(batch)
                descs[i:i + batch_size] = d.to('cpu')
                kpts[i:i + batch_size] = k.to('cpu')

        return cls(kpts, descs, batch_size=[images.size(0)])


if __name__ == '__main__':
    gen = torch.manual_seed(19971222)
    resize = T.Resize(256, antialias=True)

    example_image = read_image("DJI_20250321072428_0040_D.JPG", ImageReadMode.RGB)
    example_image = resize(example_image)
    reference_image = read_image("DJI_20250321072433_0042_D.JPG", ImageReadMode.RGB)
    reference_image = resize(reference_image)

    index = KeyPoints.from_images(example_image.unsqueeze(0), batch_size=1)
    index_kps_image = utils.visualize_keypoints(example_image, index.pts)

    query = KeyPoints.from_images(reference_image.unsqueeze(0), batch_size=1)
    query_kps_image = utils.visualize_keypoints(reference_image, query.pts)

    side_by_side = utils.concat(index_kps_image, query_kps_image, dim=1)

    tgt_ind, valid = ratio_test_threshold_match(index.desc, query.desc, 0.4)
    src_kpts = index.pts
    tgt_kpts = query.pts.gather(dim=-2, index=tgt_ind.unsqueeze(-1).expand_as(query.pts))
    pts1 = torch.cat([src_kpts, torch.ones_like(src_kpts[..., [0]])], dim=-1)
    pts2 = torch.cat([tgt_kpts, torch.ones_like(tgt_kpts[..., [0]])], dim=-1)
    errs, inliners = ransac(pts1, pts2, valid.float(), 0.75, 512)
    inliners = inliners & valid

    print(len(inliners))

    index_pts = src_kpts[inliners]
    query_pts = tgt_kpts[inliners]

    lines_with_ransac = utils.draw_match_lines(
        example_image, reference_image, index_pts, query_pts
    )

    H, mask = kornia.geometry.find_homography_dlt(index_pts.unsqueeze(0), query_pts.unsqueeze(0))

    def pad_and_warp(img, H, size):
        # Pad to 2x size to prevent cut-off
        pad_img = F.pad(img, (256, 256, 256, 256), fill=0)
        pad_img = pad_img.unsqueeze(0).float() / 255.0  # BxCxHxW normalized
        warped = kornia.geometry.warp_perspective(pad_img, H, dsize=size, mode='bilinear', padding_mode='zeros')
        return warped.squeeze(0)


    warped_ref = pad_and_warp(reference_image, H, size=(512, 512))

    padded_example = F.pad(example_image, (256, 256, 256, 256), fill=0)
    padded_example = padded_example.float() / 255.0

    merged = torch.where(warped_ref > 0, (warped_ref + padded_example) / 2, padded_example)

    plt.figure(figsize=(12, 12))
    plt.imshow(merged.permute(1, 2, 0).clip(0, 1).numpy())
    plt.title("Stitched Image")
    plt.axis("off")
    plt.show()
