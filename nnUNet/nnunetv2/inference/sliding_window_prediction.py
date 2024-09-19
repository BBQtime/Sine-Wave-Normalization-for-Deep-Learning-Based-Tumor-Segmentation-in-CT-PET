from functools import lru_cache

import numpy as np
import torch
from typing import Union, Tuple, List
from acvl_utils.cropping_and_padding.padding import pad_nd_image
from scipy.ndimage import gaussian_filter


@lru_cache(maxsize=2)
def compute_gaussian(tile_size: Union[Tuple[int, ...], List[int]], sigma_scale: float = 1. / 8,
                     value_scaling_factor: float = 1, dtype=torch.float16, device=torch.device('cuda', 0)) \
        -> torch.Tensor:
    tmp = np.zeros(tile_size)
    center_coords = [i // 2 for i in tile_size]
    sigmas = [i * sigma_scale for i in tile_size]
    tmp[tuple(center_coords)] = 1
    gaussian_importance_map = gaussian_filter(tmp, sigmas, 0, mode='constant', cval=0)

    gaussian_importance_map = torch.from_numpy(gaussian_importance_map)

    gaussian_importance_map /= (torch.max(gaussian_importance_map) / value_scaling_factor)
    gaussian_importance_map = gaussian_importance_map.to(device=device, dtype=dtype)
    # gaussian_importance_map cannot be 0, otherwise we may end up with nans!
    mask = gaussian_importance_map == 0
    gaussian_importance_map[mask] = torch.min(gaussian_importance_map[~mask])
    return gaussian_importance_map


# def compute_steps_for_sliding_window(image_size: Tuple[int, ...], tile_size: Tuple[int, ...], tile_step_size: float) -> \
#         List[List[int]]:
#     assert [i >= j for i, j in zip(image_size, tile_size)], "image size must be as large or larger than patch_size"
#     assert 0 < tile_step_size <= 1, 'step_size must be larger than 0 and smaller or equal to 1'

#     # our step width is patch_size*step_size at most, but can be narrower. For example if we have image size of
#     # 110, patch size of 64 and step_size of 0.5, then we want to make 3 steps starting at coordinate 0, 23, 46
#     target_step_sizes_in_voxels = [i * tile_step_size for i in tile_size]

#     num_steps = [int(np.ceil((i - k) / j)) + 1 for i, j, k in zip(image_size, target_step_sizes_in_voxels, tile_size)]

#     steps = []
#     for dim in range(len(tile_size)):
#         # the highest step value for this dimension is
#         max_step_value = image_size[dim] - tile_size[dim]
#         if num_steps[dim] > 1:
#             actual_step_size = max_step_value / (num_steps[dim] - 1)
#         else:
#             actual_step_size = 99999999999  # does not matter because there is only one step at 0

#         steps_here = [int(np.round(actual_step_size * i)) for i in range(num_steps[dim])]

#         steps.append(steps_here)

#     return steps

def compute_steps_for_sliding_window(image_size: Tuple[int, ...], tile_size: Tuple[int, ...], tile_step_size: float) -> List[List[int]]:
    assert [i >= j for i, j in zip(image_size, tile_size)], "image size must be as large or larger than tile_size"
    assert 0 < tile_step_size <= 1, 'tile_step_size must be larger than 0 and smaller or equal to 1'

    # our step width is tile_size*tile_step_size at most, but can be narrower. For example if we have image size of
    # 110, patch size of 64 and tile_step_size of 0.5, then we want to make 3 steps starting at coordinate 0, 23, 46
    num_tiles = 3 # number of tiles for x and y (coronal and sagittal in most cases)
    
    target_step_sizes_in_voxels = [i * tile_step_size for i in tile_size]
    print("image_size:", image_size)
    print("tile_size:", tile_size)
    print("tile_step_size:", tile_step_size)

    num_steps = [int(np.ceil((i - k) / j)) + 1 for i, j, k in zip(image_size, target_step_sizes_in_voxels, tile_size)]
    steps = []
    for dim in range(len(tile_size)):
        # the highest step value for this dimension is
        if dim ==0:
            max_step_value = image_size[dim] - tile_size[dim]
            if num_steps[dim] > 1:
                actual_step_size = max_step_value / (num_steps[dim] - 1)
            else:
                actual_step_size = 99999999999  # does not matter because there is only one step at 0

            steps_here = [int(np.round(actual_step_size * i)) for i in range(num_steps[dim])]
        if dim >0:
            # For dim > 0, center the tiles and pick the most centered tiles first
            step_offsets = calculate_centered_steps(image_size[dim], tile_size[dim], tile_step_size, num_tiles)
            coverage = calculate_coverage(step_offsets, tile_size[dim], image_size[dim])
            print("start coverage", coverage)
            # If coverage is less than 75%, increase the step size
            while coverage < 0.6 :
                tile_step_size += 0.1  # Increase step size
                
                step_offsets = calculate_centered_steps(image_size[dim], tile_size[dim], tile_step_size, num_tiles)
                coverage = calculate_coverage(step_offsets, tile_size[dim], image_size[dim])
                print("new coverage, and tile_step_size:", coverage, tile_step_size)
                if tile_step_size > 0.9:
                    break
            steps_here = step_offsets

        steps.append(steps_here)
        print(steps)
    return steps

def calculate_centered_steps(image_dim_size: int, tile_dim_size: int, tile_step_size: float, num_tiles: int) -> List[int]:
    center = image_dim_size // 2
    step_offsets = [int(center - (tile_dim_size // 2 + tile_step_size * tile_dim_size * (i - (num_tiles // 2)))) for i in range(num_tiles)]
    step_offsets = sorted(list(set(step_offsets)))  # Remove duplicates and sort
    return [offset for offset in step_offsets if offset >= 0 and offset + tile_dim_size <= image_dim_size]

def calculate_coverage(step_offsets: List[int], tile_dim_size: int, image_dim_size: int) -> float:
    if not step_offsets:
        return 0
    min_offset = min(step_offsets)
    max_offset = max(step_offsets) + tile_dim_size
    covered_range = max_offset - min_offset
    return covered_range / image_dim_size

if __name__ == '__main__':
    a = torch.rand((4, 2, 32, 23))
    a_npy = a.numpy()

    a_padded = pad_nd_image(a, new_shape=(48, 27))
    a_npy_padded = pad_nd_image(a_npy, new_shape=(48, 27))
    assert all([i == j for i, j in zip(a_padded.shape, (4, 2, 48, 27))])
    assert all([i == j for i, j in zip(a_npy_padded.shape, (4, 2, 48, 27))])
    assert np.all(a_padded.numpy() == a_npy_padded)
