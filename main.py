from PIL import Image

import utils, sift
import kornia.feature as KF
import kornia as K
from jit_ransac import ransac, ratio_test_threshold_match
from torchvision.io import read_image, ImageReadMode
import torchvision.transforms.v2 as T
import torch
from tqdm.auto import tqdm
from tensordict import tensorclass, MemoryMappedTensor
import matplotlib.pyplot as plt

import kornia
import torchvision.transforms.functional as F

from kornia.contrib import ImageStitcher


@tensorclass
class KeyPoints:
    pts: torch.Tensor
    desc: torch.Tensor

    @classmethod
    def from_images(cls, images: torch.Tensor, num_features=512, patch_size=41, angle_bins=8, spatial_bins=8,
                    batch_size: int = 8, device: str = "cpu"):
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


def pad_image(img, pad_factor=2):
    h, w = img.shape[-2:]
    pad_h, pad_w = h * (pad_factor - 1) // 2, w * (pad_factor - 1) // 2
    return F.pad(img, [pad_w, pad_w, pad_h, pad_h], fill=0)


def pad_and_warp(img, H, out_size):
    img = img.unsqueeze(0).float() / 255.0  # BxCxHxW
    warped = kornia.geometry.warp_perspective(img, H, dsize=out_size, mode='bilinear', padding_mode='zeros')
    return warped.squeeze(0)

def auto_crop(image, threshold=0.01):
    gray = image.mean(dim=0)

    mask = gray > threshold

    nonzero = mask.nonzero(as_tuple=False)
    if nonzero.numel() == 0:
        return image

    y_min, x_min = nonzero.min(dim=0).values
    y_max, x_max = nonzero.max(dim=0).values

    cropped = image[:, y_min:y_max+1, x_min:x_max+1]
    return cropped



if __name__ == '__main__':
    resize = T.Resize(256, antialias=True)

    matcher = KF.LoFTR(pretrained='outdoor')
    IS = ImageStitcher(matcher, estimator='ransac').cuda()

    imgs = [
        resize(read_image("DJI_20250317094745_0001_D.JPG", ImageReadMode.RGB)),
        resize(read_image("DJI_20250317094746_0002_D.JPG", ImageReadMode.RGB))
    ]

    input_dict = {
        "image0": K.color.rgb_to_grayscale(imgs[0]),  # LofTR works on grayscale images only
        "image1": K.color.rgb_to_grayscale(imgs[1]),
    }

    with torch.no_grad():
        correspondences = matcher(imgs)

    # gen = torch.manual_seed(19971222)
    #
    # transforms = T.Compose(
    #     [
    #         T.Resize(256, antialias=True),
    #         T.RandomRotation(30, interpolation=T.InterpolationMode.BILINEAR),
    #         T.RandomAffine(0, translate=(0.2, 0.2), interpolation=T.InterpolationMode.BILINEAR),
    #         T.RandomPerspective(distortion_scale=0.5),
    #         T.PILToTensor(),
    #     ]
    # )
    #
    # example_image = read_image("DJI_20250317094745_0001_D.JPG", ImageReadMode.RGB)
    # reference_image = read_image("DJI_20250317094746_0002_D.JPG", ImageReadMode.RGB)
    #
    # example_image = resize(example_image)
    # reference_image = resize(reference_image)
    #
    # example_image = pad_image(example_image, 2)
    # reference_image = pad_image(reference_image, 2)
    #
    # index = KeyPoints.from_images(example_image.unsqueeze(0), batch_size=1)
    # index_kps_image = utils.visualize_keypoints(example_image, index.pts)
    #
    # query = KeyPoints.from_images(reference_image.unsqueeze(0), batch_size=1)
    # query_kps_image = utils.visualize_keypoints(reference_image, query.pts)
    #
    # side_by_side = utils.concat(index_kps_image, query_kps_image, dim=1)
    #
    # plt.imshow(side_by_side)
    # plt.show()
    #
    # tgt_ind, valid = ratio_test_threshold_match(index.desc, query.desc, 0.75)
    # src_kpts = index.pts
    # tgt_kpts = query.pts.gather(dim=-2, index=tgt_ind.unsqueeze(-1).expand_as(query.pts))
    # pts1 = torch.cat([src_kpts, torch.ones_like(src_kpts[..., [0]])], dim=-1)
    # pts2 = torch.cat([tgt_kpts, torch.ones_like(tgt_kpts[..., [0]])], dim=-1)
    # errs, inliners = ransac(pts1, pts2, valid.float(), 0.4, 512)
    # inliners = inliners & valid
    #
    # index_pts = src_kpts[inliners]
    # query_pts = tgt_kpts[inliners]
    #
    # lines_with_ransac = utils.draw_match_lines(
    #     example_image, reference_image, index_pts, query_pts
    # )


