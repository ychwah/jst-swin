from typing import Tuple, Union
import torch
import torch.nn.functional as F
from math import ceil


class PatchOperator:

    def __init__(self, original_size: Tuple[int, int], window_size: Union[Tuple[int, int], int],
                 stride: Union[Tuple[int, int], int]):

        if isinstance(window_size, int):
            window_size = (window_size, window_size)
        if isinstance(stride, int):
            stride = (stride, stride)

        self.original_size = original_size
        self.window_size = window_size
        self.stride = stride

        self.padding = self._compute_padding()
        self.normalization_map = self._get_interpolation_map()
        self.after_padding_shape = (
            original_size[0] + self.padding[2] + self.padding[3], original_size[1] + self.padding[0] + self.padding[1])

    def _compute_padding(self) -> Tuple[int, int, int, int]:
        remainder_vertical = (self.original_size[0] - self.window_size[0]) % self.stride[0]
        remainder_horizontal = (self.original_size[1] - self.window_size[1]) % self.stride[1]
        if remainder_vertical != 0:
            vertical_padding = self.stride[0] - remainder_vertical
        else:
            vertical_padding = 0

        if remainder_horizontal != 0:
            horizontal_padding = self.stride[1] - remainder_horizontal
        else:
            horizontal_padding = 0

        if vertical_padding % 2 == 0:
            top_padding = bottom_padding = vertical_padding // 2
        else:
            top_padding = vertical_padding // 2
            bottom_padding = ceil(vertical_padding / 2)

        if horizontal_padding % 2 == 0:
            left_padding = right_padding = horizontal_padding // 2
        else:
            left_padding = horizontal_padding // 2
            right_padding = ceil(horizontal_padding / 2)
        # the new implementation with unfolding requires symmetric padding
        padding = (int(top_padding), int(bottom_padding), int(left_padding), int(right_padding))
        return padding

    def patchify(self, input_tensor) -> torch.Tensor:
        if not torch.is_tensor(input_tensor):
            raise TypeError(f"Input input type is not a Tensor. Got {type(input_tensor)}")

        if len(input_tensor.shape) != 4:
            raise ValueError(f"Invalid input shape, we expect BxCxHxW. Got: {input_tensor.shape}")

        # check if the window sliding over the image will fit into the image
        # torch's unfold drops the final patches that don't fit
        input_tensor = F.pad(input_tensor, self.padding, "reflect")

        batch_size, num_channels = input_tensor.shape[:2]
        dims = range(2, input_tensor.dim())
        for dim, patch_size, s in zip(dims, self.window_size, self.stride):
            input_tensor = input_tensor.unfold(dim, patch_size, s)
        input_tensor = input_tensor.permute(0, *dims, 1, *(dim + len(dims) for dim in dims)).contiguous()
        input_tensor = input_tensor.view(batch_size, -1, num_channels, *self.window_size)
        input_tensor.squeeze_(0)
        return input_tensor

    def _get_interpolation_map(self):
        b, c, n, m = (1, 1, self.original_size[0], self.original_size[1])
        interpolation_tensor = self.patchify(torch.ones((b, c, n, m)))

        i = 0
        j = 0
        x_lim = n + self.padding[2] + self.padding[3]
        y_lim = m + self.padding[0] + self.padding[1]
        end_i = (x_lim - self.window_size[0]) // self.stride[0]
        end_j = (y_lim - self.window_size[1]) // self.stride[1]
        overlap_i = self.window_size[0] - self.stride[0]
        overlap_j = self.window_size[1] - self.stride[1]

        # premake a normal patch to avoid for loops later
        center_patch = torch.ones((c, self.window_size[0], self.window_size[1]), dtype=torch.float32)
        for z in range(overlap_i):
            for y in range(self.window_size[1]):
                center_patch[:, z, y] *= (z + 1.0) / (overlap_i + 1.0)
                center_patch[:, self.window_size[0] - z - 1, y] *= (z + 1.0) / (overlap_i + 1.0)
            for x in range(self.window_size[0]):
                center_patch[:, x, z] *= (z + 1.0) / (overlap_j + 1.0)
                center_patch[:, x, self.window_size[1] - z - 1] *= (z + 1.0) / (overlap_j + 1.0)

        for k in range(interpolation_tensor.shape[0]):
            if j > end_j:
                j = 0
                i += 1
            orientation = (i != 0, i != end_i, j != 0, j != end_j)
            j += 1
            if orientation == (True, True, True, True):
                interpolation_tensor[k, ...] = center_patch
            else:
                # TOP OVERLAP
                if orientation[0]:
                    for z in range(overlap_i):
                        for y in range(self.window_size[1]):
                            interpolation_tensor[k, :, z, y] *= (z + 1.0) / (overlap_i + 1.0)

                # BOTTOM
                if orientation[1]:
                    for z in range(overlap_i):
                        for y in range(self.window_size[1]):
                            interpolation_tensor[k, :, self.window_size[0] - z - 1, y] *= (z + 1.0) / (overlap_i + 1.0)
                # LEFT
                if orientation[2]:
                    for z in range(overlap_j):
                        for x in range(self.window_size[0]):
                            interpolation_tensor[k, :, x, z] *= (z + 1.0) / (overlap_j + 1.0)

                # RIGHT
                if orientation[3]:
                    for z in range(overlap_i):
                        for x in range(self.window_size[0]):
                            interpolation_tensor[k, :, x, self.window_size[1] - z - 1] *= (z + 1.0) / (overlap_j + 1.0)

        interpolation_tensor.unsqueeze_(0)
        return interpolation_tensor

    def recover_from_patch(self, patches: torch.Tensor) -> torch.Tensor:
        patches = patches.unsqueeze(0)
        if patches.ndim != 5:
            raise ValueError(f"Invalid input shape, we expect BxNxCxHxW. Got: {patches.shape}")

        if (self.stride[0] > self.window_size[0]) | (self.stride[1] > self.window_size[1]):
            raise AssertionError(
                f"Stride={self.stride} should be less than or equal to Window size={self.window_size}, information is missing"
            )

        # renormalization step
        patches = patches * self.normalization_map

        patches = patches.permute(0, 2, 3, 4, 1)
        patches = patches.reshape(patches.shape[0], -1, patches.shape[-1])
        output_tensor = F.fold(input=patches, output_size=self.after_padding_shape, kernel_size=self.window_size,
                               stride=self.stride)

        # cutoff excess
        output_tensor = output_tensor[..., self.padding[2]:self.padding[2] + self.original_size[0],
                        self.padding[0]:self.padding[0] + self.original_size[1]]

        return output_tensor