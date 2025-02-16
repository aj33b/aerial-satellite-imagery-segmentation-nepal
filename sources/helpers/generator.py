import os

import numpy as np
import tifffile as tiff
import torch
from torch.utils.data import Dataset


class NepalDataset(Dataset):
    def __init__(self, data_path, split="train", transform=None):
        self.split_path = os.path.join(data_path, split)  # Path for split (e.g., train, val, test)
        self.images_dir = os.path.join(self.split_path, "images")  # Subdirectory for images
        self.masks_dir = os.path.join(self.split_path, "masks")  # Subdirectory for masks
        self.transform = transform

        # Load all image filenames and construct corresponding mask filenames
        self.image_list = os.listdir(self.images_dir)
        self.mask_list = []
        for image_name in self.image_list:
            basename, ext = os.path.splitext(image_name)
            mask_name = f"{basename}_mask{ext}"  # Append '_mask' to the image basename
            self.mask_list.append(mask_name)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        # Get the image and corresponding mask paths
        img_name = os.path.join(self.images_dir, self.image_list[idx])
        basename, ext = os.path.splitext(self.image_list[idx])
        mask_name = os.path.join(
            self.masks_dir,
            f"{basename}_mask{ext}"  # Use the new mask naming convention
        )

        # Load image and mask using tifffile
        img = tiff.imread(img_name)
        mask = tiff.imread(mask_name)

        # Fix input channels to ensure it is RGB
        if len(img.shape) == 3 and img.shape[-1] == 4:  # Drop the extra alpha channel if present
            img = img[:, :, :3]

        # Convert image and mask to uint8 if needed
        if img.dtype != np.uint8:
            img = (img * 255).astype(np.uint8)
        if mask.dtype != np.uint8:
            mask = (mask * 255).astype(np.uint8)

        # Apply transforms (if provided)
        if self.transform is not None:
            img = self.transform(img)
            mask = self.transform(mask)

        return img, mask

    def get_mask_path(self, idx):
        """
        Get the mask path for the corresponding image by index.

        Args:
            idx (int): Index of the image in the dataset.

        Returns:
            str: Full path to the corresponding mask file.
        """
        basename, ext = os.path.splitext(self.image_list[idx])
        mask_filename = f"{basename}_mask{ext}"  # Append '_mask' to the basename
        mask_path = os.path.join(self.masks_dir, mask_filename)
        return mask_path

class NepalDataGenerator:
    def __init__(self, dataset, batch_size=16, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.current_idx = 0
        self.indices = np.arange(len(dataset))
        if shuffle:
            np.random.shuffle(self.indices)

    def __len__(self):
        num_batches, remainder = divmod(len(self.dataset), self.batch_size)
        return num_batches + int(remainder > 0)
        # return len(self.dataset) // self.batch_size

    def __iter__(self):
        self.current_idx = 0
        return self

    def __next__(self):
        if self.current_idx >= len(self):
            raise StopIteration

        batch_indices = self.indices[self.current_idx * self.batch_size:(self.current_idx + 1) * self.batch_size]

        batch_images = []
        batch_masks = []

        for idx in batch_indices:
            image, mask = self.dataset[idx]
            batch_images.append(image)

            # Ensure the mask is squeezed to remove channel dimensions
            if len(mask.shape) == 3:  # Handle (H, W, 1) or (1, H, W)
                mask = mask.squeeze()  # This ensures masks are always (H, W)
            batch_masks.append(mask)

        self.current_idx += 1

        batch_images = torch.stack(batch_images)
        batch_masks = torch.stack(batch_masks)

        return batch_images, batch_masks
