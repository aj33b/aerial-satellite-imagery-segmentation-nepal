import os
import random
import shutil

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from osgeo import gdal, ogr
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from sources.helpers.logger import LoggerHelper

class UtilsHelper:
    def __init__(self):
        self.logger = LoggerHelper(logger_name="UtilsHelper").logger

    def __visualize(self, image, mask):
        print(f"Image shape: {image.shape}")
        print(f"Mask shape: {mask.shape}")
        # Select bands 3, 2, and 1 for the RGB image
        rgb_image = image[:, :, [1, 2, 0]]

        # Plot the RGB image
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))

        axs[0].imshow(image)
        axs[0].set_title("Image")

        # Plot the mask
        axs[1].imshow(mask, cmap="gray")
        axs[1].set_title("Mask")

        # Show the plot
        plt.show()

    def __generate_patch_coordinates(self, image_width, image_height, patch_size, stride):
        for x in range(0, image_width - patch_size + 1, stride):
            for y in range(0, image_height - patch_size + 1, stride):
                yield x, y

    def __merge_split_files(self, split_dirs, source_subfolder, output_dir):
        for split_dir in split_dirs:
            source_dir = os.path.join(split_dir, source_subfolder)

            if os.path.exists(source_dir):
                for fname in os.listdir(source_dir):
                    source_path = os.path.join(source_dir, fname)
                    destination_path = os.path.join(output_dir, fname)

                    if os.path.exists(destination_path):  # Handle duplicates
                        self.logger.warning(
                            f"File already exists in the merged directory and won't be duplicated: {destination_path}")
                    else:
                        shutil.move(source_path, destination_path)

    def __move_files_to_split(self, file_paths, target_dir, subfolder_name):
        for file_path in file_paths:
            filename = os.path.basename(file_path)
            destination_path = os.path.join(target_dir, subfolder_name, filename)
            if os.path.exists(destination_path):  # Handle potential overwrite issues
                self.logger.warning(f"File already exists and won't be overwritten: {destination_path}")
            else:
                shutil.move(file_path, destination_path)

    def __save_image_patch(self, patch, patch_path, image_meta):
        # Create directories if they don't exist
        os.makedirs(os.path.dirname(patch_path), exist_ok=True)

        # Save patch as TIFF image
        with rasterio.open(patch_path, 'w', **image_meta) as dst:
            dst.write(patch)

    def __save_mask_patch(self, patch, patch_path, mask_meta):
        # Create directories if they don't exist
        os.makedirs(os.path.dirname(patch_path), exist_ok=True)
        #  (after coorection)

        # Save patch as TIFF image
        with rasterio.open(patch_path, 'w', **mask_meta) as dst:
            dst.write(patch)

    def rasterize_masks(self,image_file_path, mask_shape_file_path, rasterized_dir):
        os.makedirs(rasterized_dir, exist_ok=True)
        # GDAL format and data type
        gdalformat = "GTiff"
        datatype = gdal.GDT_Float32
        # Get the mission name from the image file path
        rasterized_file_name = os.path.basename(mask_shape_file_path).replace(".shp", "")

        # Define the output rasterized file path
        rasterized_file_path = os.path.join(rasterized_dir, f"{rasterized_file_name}_rasterized.tif")

        try:
            # Get projection info from reference image
            Image = gdal.Open(image_file_path, gdal.GA_ReadOnly)
            if Image is None:
                self.logger.error(f"Failed to open image file: {image_file_path}")
                return

            # Open Shapefile
            Shapefile = ogr.Open(mask_shape_file_path)
            if Shapefile is None:
                self.logger.error(f"Failed to open shapefile: {mask_shape_file_path}")
                return
            Shapefile_layer = Shapefile.GetLayer()

            # Rasterize
            self.logger.info(f"Rasterizing shapefile for {rasterized_file_name}")

            Output = gdal.GetDriverByName(gdalformat).Create(
                rasterized_file_path, Image.RasterXSize, Image.RasterYSize, 1,
                datatype, options=["COMPRESS=DEFLATE"]
            )
            Output.SetProjection(Image.GetProjectionRef())
            Output.SetGeoTransform(Image.GetGeoTransform())

            # Write data to band 1
            Band = Output.GetRasterBand(1)
            Band.SetNoDataValue(0)

            gdal.RasterizeLayer(Output, [1], Shapefile_layer, options=["ATTRIBUTE=Class_id"])

            self.logger.info(
                f"Rasterization of shapefile for {rasterized_file_name} completed. Output saved to {rasterized_file_path}")

        except Exception as e:
            self.logger.exception(f"An error occurred while processing {rasterized_file_name}: {e}")

        finally:
            # Ensure datasets are closed
            if Band is not None:
                Band = None
            if Output is not None:
                Output = None
            if Image is not None:
                Image = None
            if Shapefile is not None:
                Shapefile = None

    def create_patches_categorical(self,image_path, mask_path, output_dir, patch_size, stride,
                                   min_mask_coverage=0.3):
        # Open satellite image
        with rasterio.open(image_path, "r") as src:
            image = src.read()
            image_meta = src.meta
            image_crs = src.crs

        # Open rasterized mask
        with rasterio.open(mask_path, "r") as src:
            mask_data = src.read(1)  # Read the first band only (assumes single-band mask)
            mask_meta = src.meta
            mask_crs = src.crs

        # Initialize counts
        processed_patches = 0
        no_of_patches_saved = 0

        # Base filenames for tracking
        image_base_name = os.path.splitext(os.path.basename(image_path))[0]
        mask_base_name = os.path.splitext(os.path.basename(mask_path))[0]

        # Calculate total potential patches
        total_patches = ((image.shape[1] - patch_size) // stride + 1) * ((image.shape[2] - patch_size) // stride + 1)

        # self.logger.info(
        #     f"Generating patches for image "{os.path.basename(image_path)}" and mask "{os.path.basename(mask_path)}"")
        # self.logger.info(f"Total patches to process: {total_patches}")

        # Initialize tqdm progress bar for the current image
        with tqdm(total=total_patches, desc=f"Patches for {image_base_name}", unit="patch", leave=False) as pbar:
            # Generate patches
            for i, (x, y) in enumerate(self.__generate_patch_coordinates(image.shape[1], image.shape[2], patch_size, stride)):
                processed_patches += 1  # Increment processed patch counter
                image_patch_name = f"{image_base_name}_patch_{i}.jpg"
                mask_patch_name = f"{image_base_name}_patch_{i}_mask.jpg"

                # Skip invalid patches (outside bounds)
                if y + patch_size > image.shape[1] or x + patch_size > image.shape[2]:
                    pbar.update(1)
                    continue

                # Extract mask patch
                mask_patch = mask_data[y:y + patch_size, x:x + patch_size]
                # self.logger.debug(f"Processing patch {i} at coordinates ({x}, {y}) with size {mask_patch.shape}")

                # Calculate mask coverage
                mask_patch_flat = mask_patch.flatten()
                non_zero_pixels = np.count_nonzero(mask_patch_flat > 0)  # Count non-background pixels
                mask_coverage = non_zero_pixels / mask_patch_flat.size

                # Skip patches with insufficient mask coverage
                if mask_coverage < min_mask_coverage:
                    # self.logger.warning(f"Skipping patch {i}: Mask coverage too low ({mask_coverage:.2f})")
                    pbar.update(1)
                    continue

                # Create class label mask
                categorical_mask = mask_patch  # Class label mask directly uses values from `mask_patch`

                # Get patch from image (only if mask coverage is sufficient)
                image_patch = image[:, y:y + patch_size, x:x + patch_size]

                # Create output paths
                patch_path = os.path.join(output_dir, "images", image_patch_name)
                patch_mask_path = os.path.join(output_dir, "masks", mask_patch_name)

                # Save image patch
                bin_image_meta = image_meta.copy()
                bin_image_meta.update({"count": image_patch.shape[0]})  # Number of channels in the image
                bin_image_meta.update({"height": image_patch.shape[1]})
                bin_image_meta.update({"width": image_patch.shape[2]})
                self.__save_image_patch(image_patch, patch_path, bin_image_meta)
                # self.logger.info(f"Image patch saved at: {patch_path}")

                # Save class label mask
                bin_mask_meta = mask_meta.copy()
                bin_mask_meta.update({"count": 1})  # Single layer for class label mask
                bin_mask_meta.update({"height": categorical_mask.shape[0]})
                bin_mask_meta.update({"width": categorical_mask.shape[1]})
                self.__save_mask_patch(categorical_mask[np.newaxis, :, :], patch_mask_path, bin_mask_meta)
                # self.logger.info(f"Mask patch saved at: {patch_mask_path}")

                no_of_patches_saved += 1
                # Update inner tqdm progress bar
                pbar.update(1)

        # Final logging per image-mask pair
        self.logger.info(
            f"Completed: '{image_base_name}' with {processed_patches} patches processed, {no_of_patches_saved} patches saved")

    def split_dataset(self,patch_output_dir):
        """
           Splits the dataset into training, validation, and test sets.

           Parameters:
               patch_output_dir (str): Directory containing the images and masks.

           Raises:
               ValueError: If no valid image-mask pairs are found or if the number of images does not match the number of masks.
        """

        # Set directory for the input images and masks
        input_images_dir = os.path.join(patch_output_dir, "images")
        input_masks_dir = os.path.join(patch_output_dir, "masks")

        # Directory to store the split dataset
        split_output_dir = os.path.join(patch_output_dir, "dataset_split")
        train_output_dir = os.path.join(split_output_dir, "train")
        val_output_dir = os.path.join(split_output_dir, "val")
        test_output_dir = os.path.join(split_output_dir, "test")

        # Gather all image files and their corresponding masks using updated naming convention
        all_image_paths = []
        all_mask_paths = []
        unpaired_images = []
        unpaired_masks = []

        for fname in os.listdir(input_images_dir):
            if fname.endswith(".jpg"):
                mask_name = fname.replace(".jpg", "_mask.jpg")  # Ensure masks follow the naming convention
                image_path = os.path.join(input_images_dir, fname)
                mask_path = os.path.join(input_masks_dir, mask_name)

                if os.path.exists(mask_path):  # Pair only if the corresponding mask exists
                    all_image_paths.append(image_path)
                    all_mask_paths.append(mask_path)
                else:
                    # Log unpaired image
                    unpaired_images.append(image_path)

        # Log unpaired masks
        for fname in os.listdir(input_masks_dir):
            if fname.endswith("_mask.jpg"):
                image_name = fname.replace("_mask.jpg", ".jpg")
                image_path = os.path.join(input_images_dir, image_name)
                if not os.path.exists(image_path):
                    unpaired_masks.append(os.path.join(input_masks_dir, fname))

        # Print logs for unpaired files
        if unpaired_images:
            self.logger.warning(f"Unpaired images found: {len(unpaired_images)}")
            for img in unpaired_images:
                self.logger.warning(f"Unpaired image: {img}")

        if unpaired_masks:
            self.logger.warning(f"Unpaired masks found: {len(unpaired_masks)}")
            for mask in unpaired_masks:
                self.logger.warning(f"Unpaired mask: {mask}")

        # Ensure input directories contain valid paired files
        if not all_image_paths or not all_mask_paths:
            raise ValueError(
                "No valid image-mask pairs found. Check the directories for properly paired `.jpg` files.")

        # Ensure that the total number of images matches the total number of masks
        if len(all_image_paths) != len(all_mask_paths):
            raise ValueError(
                "The number of image files does not match the number of mask files. Ensure all images have corresponding masks."
            )

        # Perform train-test-validation split
        train_images, temp_images, train_masks, temp_masks = train_test_split(
            all_image_paths, all_mask_paths,
            test_size=0.4, random_state=42
        )

        # Further split validation and test sets
        val_images, test_images, val_masks, test_masks = train_test_split(
            temp_images, temp_masks,
            test_size=0.5, random_state=42
        )

        # Confirm dataset sizes
        self.logger.debug(f"Training set size: {len(train_images)}")
        self.logger.debug(f"Validation set size: {len(val_images)}")
        self.logger.debug(f"Test set size: {len(test_images)}")

        # Create necessary output directories for train, val, and test splits
        for dir_path in [train_output_dir, val_output_dir, test_output_dir]:
            os.makedirs(os.path.join(dir_path, "images"), exist_ok=True)
            os.makedirs(os.path.join(dir_path, "masks"), exist_ok=True)

        # Move split files into their respective directories
        # Training set
        self.__move_files_to_split(train_images, train_output_dir, "images")
        self.__move_files_to_split(train_masks, train_output_dir, "masks")

        # Validation set
        self.__move_files_to_split(val_images, val_output_dir, "images")
        self.__move_files_to_split(val_masks, val_output_dir, "masks")

        # Test set
        self.__move_files_to_split(test_images, test_output_dir, "images")
        self.__move_files_to_split(test_masks, test_output_dir, "masks")

        self.logger.info(f"Files have been successfully split and moved to {split_output_dir}.")

    def merge_split_dataset(self, patch_output_dir):
        # Set the directory for the split dataset
        split_output_dir = os.path.join(patch_output_dir, "dataset_split")
        train_output_dir = os.path.join(split_output_dir, "train")
        val_output_dir = os.path.join(split_output_dir, "val")
        test_output_dir = os.path.join(split_output_dir, "test")

        # Set the destination directory for the merged dataset
        merged_output_dir = os.path.join(patch_output_dir, "merged_dataset")
        output_images_dir = os.path.join(merged_output_dir, "images")
        output_masks_dir = os.path.join(merged_output_dir, "masks")

        # Create the merged directories if they don't exist
        os.makedirs(output_images_dir, exist_ok=True)
        os.makedirs(output_masks_dir, exist_ok=True)


        # List of split directories
        split_dirs = [train_output_dir, val_output_dir, test_output_dir]

        # Merge images and masks back into unified directories
        self.__merge_split_files(split_dirs, "images", output_images_dir)
        self.__merge_split_files(split_dirs, "masks", output_masks_dir)

        self.logger.info(f"Files have been successfully merged into {merged_output_dir}.")

    def visualize_random_patch(self,patch_output_dir):
        # Set the base path for the dataset_split directory
        dataset_split_dir = os.path.join(patch_output_dir, "dataset_split")

        # Choose a random subset (train, val, or test)
        subsets = ["train", "val", "test"]
        random_subset = random.choice(subsets)

        # Get paths of images and masks within the chosen subset
        subset_images_dir = os.path.join(dataset_split_dir, random_subset, "images")
        subset_masks_dir = os.path.join(dataset_split_dir, random_subset, "masks")

        # Get list of all image files in the chosen subset
        image_files = [f for f in os.listdir(subset_images_dir) if f.endswith(".jpg")]

        # Ensure the subset has valid image files
        if not image_files:
            raise ValueError(f"No image files found in the {random_subset} subset.")

        # Select a random image file
        random_image_file = random.choice(image_files)

        # Construct the full paths to the randomly selected image and corresponding mask files
        temp_image_path = os.path.join(subset_images_dir, random_image_file)
        temp_mask_path = os.path.join(subset_masks_dir, random_image_file.replace(".jpg", "_mask.jpg"))

        # Open the image and mask files using Rasterio
        with rasterio.open(temp_image_path) as src:
            image = src.read().astype(float)
            image_transform = src.transform

        with rasterio.open(temp_mask_path) as src:
            mask = src.read()
            mask_transform = src.transform

        # The image data read by Rasterio is in (bands, rows, cols) order
        # Convert the image data to (rows, cols, bands) order for visualization
        transposed_image = image.transpose((1, 2, 0))

        # Scale the image data to be between 0 and 1 for better visualization
        transposed_image -= transposed_image.min()
        transposed_image /= transposed_image.max()

        # Create side-by-side plots
        fig, ax = plt.subplots(1, 2, figsize=(15, 15))

        # Display the image
        ax[0].imshow(transposed_image)
        ax[0].set_title(f"Image ({random_subset})")

        # Display the mask
        ax[1].imshow(mask[0], cmap="gray")
        ax[1].set_title(f"Mask ({random_subset})")

        plt.tight_layout()
        plt.show()

        # Print shapes for additional information
        self.logger.debug(f"Selected subset: {random_subset}")
        self.logger.debug(f"Image shape: {image.shape}")
        self.logger.debug(f"Mask shape: {mask.shape}")

    def visualize_data_generator_output(self,selected_data_generator, no_of_images_to_show,batch_size):
        images, masks = selected_data_generator.__next__()

        if no_of_images_to_show > batch_size:
            no_of_images_to_show = batch_size

        # Visualize the specified number of images and masks from the batch
        for i in range(0, no_of_images_to_show):
            image = images[i].permute(1, 2, 0).numpy()  # Access individual image and convert to numpy array
            mask = masks[i].squeeze().numpy()  # Access individual mask and convert to numpy array
            self.__visualize(image, mask)
