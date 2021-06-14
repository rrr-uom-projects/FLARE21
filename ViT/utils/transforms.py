"""
3D transforms

https://github.com/intel-isl/DPT/blob/main/dpt/transforms.py
"""
from sys import ps1
import numpy as np
import cv2
import math
from einops import rearrange
import random
from scipy.ndimage import rotate, zoom


def apply_min_size(sample, size, image_interpolation_method=cv2.INTER_AREA):
    """Rezise the sample to ensure the given size. Keeps aspect ratio.
    Args:
        sample (dict): sample
        size (tuple): image size
    Returns:
        tuple: new size
    """
    shape = list(sample["disparity"].shape)

    if shape[0] >= size[0] and shape[1] >= size[1]:
        return sample

    scale = [0, 0]
    scale[0] = size[0] / shape[0]
    scale[1] = size[1] / shape[1]

    scale = max(scale)

    shape[0] = math.ceil(scale * shape[0])
    shape[1] = math.ceil(scale * shape[1])

    # resize
    sample["image"] = cv2.resize(
        sample["image"], tuple(shape[::-1]), interpolation=image_interpolation_method
    )

    sample["disparity"] = cv2.resize(
        sample["disparity"], tuple(shape[::-1]), interpolation=cv2.INTER_NEAREST
    )
    sample["mask"] = cv2.resize(
        sample["mask"].astype(np.float32),
        tuple(shape[::-1]),
        interpolation=cv2.INTER_NEAREST,
    )
    sample["mask"] = sample["mask"].astype(bool)

    return tuple(shape)


class Resize(object):
    """Resize sample to given size (width, height)."""

    def __init__(
        self,
        width,
        height,
        depth,
        resize_target=True,
        keep_aspect_ratio=False,
        ensure_multiple_of=1,
        resize_method="lower_bound",
        image_interpolation_method=cv2.INTER_AREA,
    ):
        """Init.
        Args:
            width (int): desired output width
            height (int): desired output height
            depth (int): desired output depth
            resize_target (bool, optional):
                True: Resize the full sample (image, mask, target).
                False: Resize image only.
                Defaults to True.
            keep_aspect_ratio (bool, optional):
                True: Keep the aspect ratio of the input sample.
                Output sample might not have the given width and height, and
                resize behaviour depends on the parameter 'resize_method'.
                Defaults to False.
            ensure_multiple_of (int, optional):
                Output width and height is constrained to be multiple of this parameter.
                Defaults to 1.
            resize_method (str, optional):
                "lower_bound": Output will be at least as large as the given size.
                "upper_bound": Output will be at max as large as the given size. (Output size might be smaller than given size.)
                "minimal": Scale as least as possible.  (Output size might be smaller than given size.)
                Defaults to "lower_bound".
        """
        self.__width = width
        self.__height = height
        self.__depth = depth

        self.__resize_target = resize_target
        self.__keep_aspect_ratio = keep_aspect_ratio
        self.__multiple_of = ensure_multiple_of
        self.__resize_method = resize_method
        self.__image_interpolation_method = image_interpolation_method

    def constrain_to_multiple_of(self, x, min_val=0, max_val=None):
        y = (np.round(x / self.__multiple_of) * self.__multiple_of).astype(int)

        if max_val is not None and y > max_val:
            y = (np.floor(x / self.__multiple_of)
                 * self.__multiple_of).astype(int)

        if y < min_val:
            y = (np.ceil(x / self.__multiple_of)
                 * self.__multiple_of).astype(int)

        return y

    def get_size(self, width, height, depth):
        # determine new height and width
        scale_height = self.__height / height
        scale_width = self.__width / width
        scale_depth = self.__depth / depth

        if self.__keep_aspect_ratio:
            if self.__resize_method == "lower_bound":
                #* scale such that output size is lower bound
                if scale_width > scale_height:
                    # fit width
                    scale_height = scale_width
                else:
                    # fit height
                    scale_width = scale_height
            elif self.__resize_method == "upper_bound":
                #* scale such that output size is upper bound
                if scale_width < scale_height:
                    # fit width
                    scale_height = scale_width
                else:
                    # fit height
                    scale_width = scale_height
            elif self.__resize_method == "minimal":
                #* scale as little as possbile
                if abs(1 - scale_width) < abs(1 - scale_height):
                    # fit width
                    scale_height = scale_width
                else:
                    # fit height
                    scale_width = scale_height
            else:
                raise ValueError(
                    f"resize_method {self.__resize_method} not implemented"
                )

        if self.__resize_method == "lower_bound":
            new_height = self.constrain_to_multiple_of(
                scale_height * height, min_val=self.__height
            )
            new_width = self.constrain_to_multiple_of(
                scale_width * width, min_val=self.__width
            )
            new_depth = self.constrain_to_multiple_of(
                scale_depth * depth, min_val = self.__depth)

        elif self.__resize_method == "upper_bound":
            new_height = self.constrain_to_multiple_of(
                scale_height * height, max_val=self.__height
            )
            new_width = self.constrain_to_multiple_of(
                scale_width * width, max_val=self.__width
            )
            new_depth = self.constrain_to_multiple_of(
                scale_depth * depth, max_val= self.__depth)
        elif self.__resize_method == "minimal":
            new_height = self.constrain_to_multiple_of(scale_height * height)
            new_width = self.constrain_to_multiple_of(scale_width * width)
            new_depth = self.constrain_to_multiple_of(scale_depth * depth)
        else:
            raise ValueError(
                f"resize_method {self.__resize_method} not implemented")

        return (new_width, new_height, new_depth)

    def __call__(self, sample):
        width, height, depth = self.get_size(
            sample["image"].shape[1], sample["image"].shape[0], sample["image"].shape[2]
        )

        # resize sample
        sample["image"] = cv2.resize(
            sample["image"],
            (width, height, depth),
            interpolation=self.__image_interpolation_method,
        )

        if self.__resize_target:
            if "disparity" in sample:
                sample["disparity"] = cv2.resize(
                    sample["disparity"],
                    (width, height, depth),
                    interpolation=cv2.INTER_NEAREST,
                )

            if "depth" in sample:
                sample["depth"] = cv2.resize(
                    sample["depth"], (width,
                                      height, depth), interpolation=cv2.INTER_NEAREST
                )

            sample["mask"] = cv2.resize(
                sample["mask"].astype(np.float32),
                (width, height, depth),
                interpolation=cv2.INTER_NEAREST,
            )
            sample["mask"] = sample["mask"].astype(bool)

        return sample

class NormalizeImage(object):
    """Normalize image by given mean and std."""

    def __init__(self, mean, std):
        self.__mean = mean
        self.__std = std

    def __call__(self, sample):
        sample["image"] = (sample["image"] - self.__mean) / self.__std

        return sample


class PrepareForNet(object):
    """Prepare sample for usage as network input."""

    def __init__(self):
        pass

    def __call__(self, sample):
        #* Convert to channels first
        image = rearrange(sample["image"], 'h w d c -> c h w d')
        sample["image"] = np.ascontiguousarray(image).astype(np.float32)

        if "mask" in sample:
            sample["mask"] = sample["mask"].astype(np.float32)
            sample["mask"] = np.ascontiguousarray(sample["mask"])

        if "disparity" in sample: #!! No idea what this is for
            disparity = sample["disparity"].astype(np.float32)
            sample["disparity"] = np.ascontiguousarray(disparity)

        if "depth" in sample: #!! Or this
            depth = sample["depth"].astype(np.float32)
            sample["depth"] = np.ascontiguousarray(depth)

        return sample

################################## -------- ED TRANSFORMS ----- ##############################

class ShiftImage():
    #@ From Ed's dataloader
    def __init__(self, range_=[2, 4, 4], p=0.5):
        self.range = range_
        self.p = p

    def __call__(self, sample):
       #!! Think this assumes channels are in last dim.
        # find shift values
        if random.random() > self.p:
            cc_shift, ap_shift, lr_shift = random.randint(-self.range[0], self.range[0]), \
                    random.randint(-self.range[1], self.range[1]), \
                    random.randint(-self.range[2], self.range[2])
            # pad for shifting into

            pad_width = ((x, x) for x in self.range)
            ct_im = np.pad(sample["image"], pad_width=pad_width,
                            mode='constant', constant_values=-1024)
            mask = np.pad(sample["mask"], pad_width=pad_width,
                            mode='constant', constant_values=0)
            # crop to complete shift
            #!! Assumes image size  = [64, 128, 128]
            sample["image"] = ct_im[2+cc_shift:66+cc_shift, 4 +
                            ap_shift:132+ap_shift, 4+lr_shift:132+lr_shift]
            sample["mask"] = mask[2+cc_shift:66+cc_shift, 4 +
                        ap_shift:132+ap_shift, 4+lr_shift:132+lr_shift]
        return sample

class RotateImage():
    #@ Also from Ed's dataloader
    def __init__(self, range_ = 10, p=0.5):
        self.range = range_
        self.p = p

    def __call__(self, sample):
        if random.random() > self.p:
            roll_angle = np.clip(np.random.normal(loc=0, scale=3), -self.range, self.range)
            sample["image"] = self.rotation(
                sample["image"], roll_angle, rotation_plane=(1, 2), is_mask=False)
            sample["mask"] = self.rotation(
                sample["mask"], roll_angle, rotation_plane=(1, 2), is_mask=True)
        
        return sample

    def rotation(self, image, rot_angle, rot_plane, is_mask):
        # rotate the image or mask using scipy rotate function
        order, cval = (0, 0) if is_mask else (3, -1024)
        return rotate(input=image, angle=rot_angle, axes=rot_plane, reshape=False, order=order, mode='constant', cval=cval)


class ScaleImage():
    #@ Also from Ed
    def __init__(self, range_=0.2, p=0.5):
        self.p = p
        self.range = range_

    def __call__(self, sample):
        if random.random() > self.p:
            scale_factor = np.clip(np.random.normal(loc=1.0, scale=0.05), 1-self.range, 1+self.range)
            sample["image"] = self.scale(sample["image"], scale_factor, is_mask=False)
            sample["mask"] = self.scale(sample["mask"], scale_factor, is_mask=True)
        return sample

    def scale(self, image, scale_factor, is_mask):
         # scale the image or mask using scipy zoom function
        order, cval = (0, 0) if is_mask else (3, -1024)
        height, width, depth = image.shape
        zheight = int(np.round(scale_factor*height))
        zwidth = int(np.round(scale_factor*width))
        zdepth = int(np.round(scale_factor*depth))
        # zoomed out
        if scale_factor < 1.0:
            new_image = np.full_like(image, cval)
            ud_buffer = (height-zheight) // 2
            ap_buffer = (width-zwidth) // 2
            lr_buffer = (depth-zdepth) // 2
            new_image[ud_buffer:ud_buffer+zheight, ap_buffer:ap_buffer+zwidth, lr_buffer:lr_buffer+zdepth] = zoom(
                input=image, zoom=scale_factor, order=order, mode='constant', cval=cval)[0:zheight, 0:zwidth, 0:zdepth]
            return new_image
        elif scale_factor > 1.0:
            new_image = zoom(input=image, zoom=scale_factor, order=order,
                             mode='constant', cval=cval)[0:zheight, 0:zwidth, 0:zdepth]
            ud_extra = (new_image.shape[0] - height) // 2
            ap_extra = (new_image.shape[1] - width) // 2
            lr_extra = (new_image.shape[2] - depth) // 2
            new_image = new_image[ud_extra:ud_extra+height,
                                  ap_extra:ap_extra+width, lr_extra:lr_extra+depth]
            return new_image
        return image
