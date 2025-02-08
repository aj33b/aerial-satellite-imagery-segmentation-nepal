import logging
import os
from logging.handlers import TimedRotatingFileHandler

import matplotlib.pyplot as plt
import numpy as np
import rasterio
import torch
from osgeo import gdal, ogr
from rasterio.features import rasterize
from shapely import Polygon
from shapely.ops import unary_union
from torchvision.transforms import transforms
from tqdm.notebook import tqdm_notebook as tqdm

from generator import NepalDataset, NepalDataGenerator

# Setup logger

class DynamicTqdmHandler(logging.StreamHandler):
    """
    Custom logging handler that dynamically switches between `tqdm.write` and standard `StreamHandler`
    based on whether a `tqdm` progress bar is currently active.
    """

    def emit(self, record):
        try:
            msg = self.format(record)
            # If tqdm is active, use tqdm.write to log below the progress bar
            if tqdm.get_lock().locks:  # Check if a tqdm instance is active
                tqdm.write(msg)
            else:
                # If tqdm is not active, fallback to standard `StreamHandler` behavior
                super().emit(record)
        except Exception:
            self.handleError(record)

class ColorFormatter(logging.Formatter):
    COLORS = {
        'DEBUG': '\033[94m',  # Blue
        'INFO': '\033[92m',  # Green
        'WARNING': '\033[93m',  # Yellow
        'ERROR': '\033[91m',  # Red
        'CRITICAL': '\033[1;91m'  # Bold Red
    }
    RESET = '\033[0m'

    def format(self, record):
        color = self.COLORS.get(record.levelname, self.RESET)
        formatted_message = super().format(record)
        return f"{color}{formatted_message}{self.RESET}"

def setup_logger(logger_name='Satellite Segmentation Nepal',
                 log_level=logging.DEBUG,
                 log_dir='./logs'):
    """
    Sets up a logger with both console and file handlers. The console handler
    dynamically uses tqdm logging when active.

    Parameters:
        logger_name (str): The name of the logger.
        log_level (int): The logging level (e.g., logging.DEBUG, logging.INFO).
        log_dir (str): Directory where log files will be saved.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(log_level)

    # Avoid duplicate handlers in case the logger is already initialized
    if logger.handlers:
        return logger

    # Create the logs directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)

    # Log format
    LOG_FORMAT = (
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s "
        "[in %(pathname)s:%(lineno)d]"
    )

    # Console handler with dynamic tqdm support
    stream_handler = DynamicTqdmHandler()
    stream_handler.setLevel(log_level)
    stream_handler.setFormatter(ColorFormatter(LOG_FORMAT))
    logger.addHandler(stream_handler)

    # Timed rotating file handler (rotates logs daily at midnight)
    file_handler = TimedRotatingFileHandler(
        os.path.join(log_dir, 'log.txt'),
        when='midnight',
        interval=1,
        backupCount=7,  # Keep logs for the last 7 days
        encoding='utf-8'
    )
    file_handler.setLevel(log_level)
    file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
    logger.addHandler(file_handler)

    return logger

def poly_from_utm(polygon, transform):
    poly_pts = []

    poly = unary_union(polygon)
    for i in np.array(poly.exterior.coords):
        poly_pts.append(~transform * tuple(i))

    new_poly = Polygon(poly_pts)
    return new_poly


"""
def create_patches(image_path, mask_path, output_dir, patch_size, stride, boundary):
    # Open satellite image
    with rasterio.open(image_path, "r") as src:
        image = src.read()
        image_meta = src.meta
        image_crs = src.crs

    # Open shapefile mask
    mask_data = gpd.read_file(mask_path)
    mask_data = mask_data.to_crs(src.crs)

    # Get area of interest boundary to cut the patches
    min_lat, min_lon, max_lat, max_lon = boundary
    no_of_patches_saved = 0

    tqdm.write("Generating patches...")

    # Iterate over patches
    for i, (x, y) in tqdm(enumerate(generate_patch_coordinates(image.shape[1], image.shape[2], patch_size, stride))):
        # Calculate patch boundary
        patch_boundary = (
            min_lon + (max_lon - min_lon) * (x / image.shape[1]),
            min_lon + (max_lon - min_lon) * ((x + patch_size) / image.shape[1]),
            min_lat + (max_lat - min_lat) * (y / image.shape[2]),
            min_lat + (max_lat - min_lat) * ((y + patch_size) / image.shape[2])
        )

        # Create patch geometry
        patch_geometry = box(patch_boundary[0], patch_boundary[2], patch_boundary[1], patch_boundary[3])

        # Check if patch boundary intersects with any mask geometry
        intersecting_masks = mask_data[mask_data.intersects(patch_geometry)]

        # Print the number of intersecting masks for debugging
        tqdm.write(f"\nNumber of intersecting masks for patch {i}: {len(intersecting_masks)}")

        # Skip patch if no intersection with mask geometries
        if intersecting_masks.empty:
            continue

        # Get patch from image
        patch = image[:, x:x + patch_size, y:y + patch_size]

        # Create patch mask by rasterizing the intersecting mask geometries
        patch_mask = rasterize_masks(image_meta, intersecting_masks, patch_size, patch_size)

        # Save patch and patch mask
        patch_name = f"patch_{i}.tiff"
        patch_mask_name = f"patch_{i}_mask.tiff"
        patch_path = os.path.join(output_dir, "images", patch_name)
        patch_mask_path = os.path.join(output_dir, "masks", patch_mask_name)

        bin_image_meta = image_meta.copy()
        bin_image_meta.update({'count': patch.shape[0]})
        bin_image_meta.update({'height': patch.shape[1]})
        bin_image_meta.update({'width': patch.shape[2]})

        save_image_patch(patch, patch_path, bin_image_meta)
        tqdm.write(f"Image patch saved at: {output_dir}/images/{patch_name}")

        bin_mask_meta = image_meta.copy()
        bin_mask_meta.update({'count': 1})
        bin_mask_meta.update({'height': patch_mask.shape[0]})
        bin_mask_meta.update({'width': patch_mask.shape[1]})

        save_mask_patch(patch_mask, patch_mask_path, bin_mask_meta)

        tqdm.write(f"Mask patch saved at: {output_dir}/images/{patch_name}")
        no_of_patches_saved += 1
        # TODO: Remove this
        if (no_of_patches_saved == 10):
            break;
    tqdm.write(f"\nTotal number image and mask patches saved: {str(no_of_patches_saved)}")

"""


def create_patches(image_path, mask_path, output_dir, patch_size, stride, logger):
    try:
        # Open satellite image
        with rasterio.open(image_path, "r") as src:
            image = src.read()
            image_meta = src.meta
            image_crs = src.crs

        # Open shapefile mask
        with rasterio.open(mask_path, "r") as src:
            mask_data = src.read()
            mask_meta = src.meta
            mask_crs = src.crs

        no_of_patches_saved = 0

        logger.info(f"Generating patches for image: {image_path} and mask: {mask_path}")

        # Extract base names for image and mask
        image_base_name = os.path.splitext(os.path.basename(image_path))[0]
        mask_base_name = os.path.splitext(os.path.basename(mask_path))[0]

        # Iterate over patches
        for i, (x, y) in enumerate(generate_patch_coordinates(image.shape[1], image.shape[2], patch_size, stride)):
            image_patch_name = f"{image_base_name}_patch_{i}.jpg"
            mask_patch_name = f"{mask_base_name}_patch_{i}_mask.jpg"

            # Check if the patch size is smaller than the remaining image size
            if y + patch_size > image.shape[1] or x + patch_size > image.shape[2]:
                continue

            mask_patch = mask_data[0, y:y + patch_size, x:x + patch_size]

            # Calculate the number of unique non-zero values in the patch
            unique_values = np.unique(mask_patch)
            num_intersecting_masks = len(np.setdiff1d(unique_values, [0]))
            logger.info(f"Number of intersecting masks for {mask_patch_name}: {num_intersecting_masks}")
            # Skip patch if no intersection with mask geometries
            if np.count_nonzero(mask_patch) == 0:
                continue

            # Get patch from image
            image_patch = image[:, y:y + patch_size, x:x + patch_size]

            # Generate file names using base names
            patch_path = os.path.join(output_dir, "images", image_patch_name)
            patch_mask_path = os.path.join(output_dir, "masks", mask_patch_name)

            bin_image_meta = image_meta.copy()
            bin_image_meta.update({'count': image_patch.shape[0]})
            bin_image_meta.update({'height': image_patch.shape[1]})
            bin_image_meta.update({'width': image_patch.shape[2]})

            save_image_patch(image_patch, patch_path, bin_image_meta)
            logger.info(f"Image patch saved at: {patch_path}")

            bin_mask_meta = mask_meta.copy()
            bin_mask_meta.update({'count': 1})
            bin_mask_meta.update({'height': mask_patch.shape[0]})
            bin_mask_meta.update({'width': mask_patch.shape[1]})

            save_mask_patch(mask_patch, patch_mask_path, bin_mask_meta)
            logger.info(f"Mask patch saved at: {patch_mask_path}")
            no_of_patches_saved += 1

        logger.info(f"Total number of image and mask patches saved: {no_of_patches_saved}")

    except Exception as e:
        logger.error(f"An error occurred while generating patches: {e}")


def create_patches_one_hot(image_path, mask_path, output_dir, patch_size, stride, logger, min_mask_coverage=0.3):
    # Open satellite image
    with rasterio.open(image_path, "r") as src:
        image = src.read()
        image_meta = src.meta
        image_crs = src.crs

    # Open rasterized mask
    with rasterio.open(mask_path, "r") as src:
        mask_data = src.read()
        mask_meta = src.meta
        mask_crs = src.crs

    no_of_patches_saved = 0
    image_base_name = os.path.splitext(os.path.basename(image_path))[0]
    mask_base_name = os.path.splitext(os.path.basename(mask_path))[0]

    logger.info(f"Generating patches for image: {image_path} and mask: {mask_path}")

    # Iterate over patches
    for i, (x, y) in enumerate(generate_patch_coordinates(image.shape[1], image.shape[2], patch_size, stride)):
        image_patch_name = f"{image_base_name}_patch_{i}.jpg"
        mask_patch_name = f"{mask_base_name}_patch_{i}_mask.jpg"

        # Check if the patch size is smaller than the remaining image size and skip if not
        if y + patch_size > image.shape[1] or x + patch_size > image.shape[2]:
            continue

        mask_patch = mask_data[:, y:y + patch_size, x: x + patch_size]
        logger.debug(f"Processing {image_base_name} patch {i} at coordinates ({x}, {y}) with size {mask_patch.shape}")

        # Calculate mask coverage
        mask_patch_flat = mask_patch.flatten()
        non_zero_pixels = np.count_nonzero(mask_patch_flat)
        mask_coverage = non_zero_pixels / mask_patch_flat.size

        # Check if mask coverage is below threshold and skip if not
        if mask_coverage < min_mask_coverage:
            logger.warning(f"Skipping {image_base_name} patch {i} - Mask coverage below threshold: {mask_coverage:.2f}")
            continue

        one_hot_mask = np.zeros((3, patch_size, patch_size))
        for class_value in [0, 1, 2]:  # Assuming classes 1 and 2
            class_indices = np.where(mask_patch == class_value, class_value + 1, 0)
            one_hot_mask[class_value, :, :] = class_indices

        # Get patch from image
        # Get patch from image (only if mask coverage is sufficient)
        image_patch = image[:, y:y + patch_size, x:x + patch_size]

        # Save patch and patch mask
        patch_path = os.path.join(output_dir, "images", image_patch_name)
        patch_mask_path = os.path.join(output_dir, "masks", mask_patch_name)

        bin_image_meta = image_meta.copy()
        bin_image_meta.update({'count': image_patch.shape[0]})
        bin_image_meta.update({'height': image_patch.shape[1]})
        bin_image_meta.update({'width': image_patch.shape[2]})

        save_image_patch(image_patch, patch_path, bin_image_meta)
        logger.info(f"Image patch saved at: {output_dir}/images/{image_patch_name}")

        bin_mask_meta = mask_meta.copy()
        bin_mask_meta.update({'count': one_hot_mask.shape[0]})
        bin_mask_meta.update({'height': one_hot_mask.shape[1]})
        bin_mask_meta.update({'width': one_hot_mask.shape[2]})

        save_mask_patch(one_hot_mask, patch_mask_path, bin_mask_meta)

        logger.info(f"Mask patch saved at: {output_dir}/images/{image_patch_name}")
        no_of_patches_saved += 1
    logger.info(f"\nTotal number of image and mask patches saved: {str(no_of_patches_saved)}")

def create_patches_categorical(image_path, mask_path, output_dir, patch_size, stride, logger, min_mask_coverage=0.3):
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

    # logger.info(
    #     f"Generating patches for image '{os.path.basename(image_path)}' and mask '{os.path.basename(mask_path)}'")
    # logger.info(f"Total patches to process: {total_patches}")

    # Initialize tqdm progress bar for the current image
    with tqdm(total=total_patches, desc=f"Patches for {image_base_name}", unit="patch", leave=False) as pbar:
        # Generate patches
        for i, (x, y) in enumerate(generate_patch_coordinates(image.shape[1], image.shape[2], patch_size, stride)):
            processed_patches += 1  # Increment processed patch counter
            image_patch_name = f"{image_base_name}_patch_{i}.jpg"
            mask_patch_name = f"{image_base_name}_patch_{i}_mask.jpg"

            # Skip invalid patches (outside bounds)
            if y + patch_size > image.shape[1] or x + patch_size > image.shape[2]:
                pbar.update(1)
                continue

            # Extract mask patch
            mask_patch = mask_data[y:y + patch_size, x:x + patch_size]
            # logger.debug(f"Processing patch {i} at coordinates ({x}, {y}) with size {mask_patch.shape}")

            # Calculate mask coverage
            mask_patch_flat = mask_patch.flatten()
            non_zero_pixels = np.count_nonzero(mask_patch_flat > 0)  # Count non-background pixels
            mask_coverage = non_zero_pixels / mask_patch_flat.size

            # Skip patches with insufficient mask coverage
            if mask_coverage < min_mask_coverage:
                # logger.warning(f"Skipping patch {i}: Mask coverage too low ({mask_coverage:.2f})")
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
            bin_image_meta.update({'count': image_patch.shape[0]})  # Number of channels in the image
            bin_image_meta.update({'height': image_patch.shape[1]})
            bin_image_meta.update({'width': image_patch.shape[2]})
            save_image_patch(image_patch, patch_path, bin_image_meta)
            # logger.info(f"Image patch saved at: {patch_path}")

            # Save class label mask
            bin_mask_meta = mask_meta.copy()
            bin_mask_meta.update({'count': 1})  # Single layer for class label mask
            bin_mask_meta.update({'height': categorical_mask.shape[0]})
            bin_mask_meta.update({'width': categorical_mask.shape[1]})
            save_mask_patch(categorical_mask[np.newaxis, :, :], patch_mask_path, bin_mask_meta)
            # logger.info(f"Mask patch saved at: {patch_mask_path}")

            no_of_patches_saved += 1
            # Update inner tqdm progress bar
            pbar.update(1)

    # Final logging per image-mask pair
    logger.info(
        f"Completed: '{image_base_name}' with {processed_patches} patches processed, {no_of_patches_saved} patches saved")

def visualize_image_mask(image_array, mask_array):
    # Check if the image is 5-channel
    if image_array.shape[-1] == 5:
        # Create a figure with 6 subplots (5 for image channels and 1 for the mask)
        fig, axes = plt.subplots(1, 6, figsize=(15, 5))
        fig.suptitle('Image and Mask Visualization')

        # Display each channel of the image in separate subplots
        for i in range(5):
            axes[i].imshow(image_array[:, :, i], cmap='gray')
            axes[i].set_title(f'Channel {i + 1}')
            axes[i].axis('off')
    else:
        # If the image is not 5-channel, display it as a single subplot
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        fig.suptitle('Image and Mask Visualization')

        axes[0].imshow(image_array, cmap='gray')
        axes[0].set_title('Image')
        axes[0].axis('off')

    # Display the mask in the last subplot
    axes[-1].imshow(mask_array, cmap='gray')
    axes[-1].set_title('Mask')
    axes[-1].axis('off')

    # Show the plots
    plt.show()


def visualize(image, mask):

    print(f"Image shape: {image.shape}")
    print(f"Mask shape: {mask.shape}")
    # Select bands 3, 2, and 1 for the RGB image
    rgb_image = image[:, :, [1, 2, 0]]

    # Plot the RGB image
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    axs[0].imshow(image)
    axs[0].set_title("Image")

    # Plot the mask
    axs[1].imshow(mask, cmap='gray')
    axs[1].set_title("Mask")

    # Show the plot
    plt.show()


def generate_patch_coordinates(image_width, image_height, patch_size, stride):
    for x in range(0, image_width - patch_size + 1, stride):
        for y in range(0, image_height - patch_size + 1, stride):
            yield x, y


""" 
def rasterize_masks(intersecting_masks, patch_geometry, image_width, image_height,transform):
    # Create mask image with same dimensions as the patch
    mask_image = np.zeros((1, image_width, image_height), dtype=np.uint8)

    # Iterate over intersecting mask geometries
    for idx, mask_geometry in intersecting_masks.iterrows():
        # Clip mask geometry with patch geometry
        clipped_geometry = mask_geometry.geometry.intersection(patch_geometry)

        # if clipped_geometry.is_empty:
        #     print(f"Empty geometry for index {idx}. Skipping...")
        #     continue

        # # Print clipped geometry and patch_geometry for debugging
        # print(f"Clipped Geometry for index {idx}:\n", clipped_geometry)
        # print(f"Patch Geometry for index {idx}:\n", patch_geometry)

        # Rasterize clipped geometry into the mask image
        mask_array = geometry_mask([clipped_geometry],
                                   out_shape=(image_width, image_height),
                                   transform=transform,
                                   invert=True)

        # Combine the mask_array with the mask_image using OR operation
        mask_image |= mask_array.astype(np.uint8)

        # Print the mask_array for debugging
        print(f"Mask Array for index {idx}:\n", mask_array)

    return mask_image

 """

"""
def rasterize_masks(intersecting_masks, patch_geometry, image_width, image_height, transform, num_classes=22):
    # Create an empty numpy array to store the mask
    mask = np.zeros((num_classes, image_height, image_width), dtype=np.uint8)

    # Iterate over each class in the intersecting masks
    for i, class_name in enumerate(intersecting_masks['Class'].unique()):
        # Get the subset of data for the current class
        class_data = intersecting_masks[intersecting_masks['Class'] == class_name]

        # Convert the class data to a list of geometries
        class_geoms = list(class_data['geometry'])

        # Rasterize the class geometries to a binary mask
        class_mask = geometry_mask(
            class_geoms,
            out_shape=(image_height, image_width),
            transform=transform,
            invert=True
        )

        # Store the class mask in the output mask array
        mask[i] = class_mask.astype(np.uint8)

    return mask
"""

"""
def rasterize_masks(image_meta, intersecting_masks, image_height, image_width):
    poly_shp = []
    im_size = (image_height, image_width)
    for num, row in intersecting_masks.iterrows():
        if row['geometry'].geom_type == 'Polygon':
            poly = poly_from_utm(row['geometry'], image_meta['transform'])
            poly_shp.append(poly)
        else:
            for p in row['geometry']:
                poly = poly_from_utm(p, image_meta['transform'])
                poly_shp.append(poly)

    mask = rasterize(shapes=poly_shp,
                     out_shape=im_size, transform=image_meta['transform'], )
    mask = mask.astype("uint16")

    return mask
"""

def rasterize_masks(image_file_path, mask_shape_file_path,rasterized_dir,logger):
    # GDAL format and data type
    gdalformat = 'GTiff'
    datatype = gdal.GDT_Float32
    try:
        # Get the mission name from the image file path
        rasterized_file_name = os.path.basename(mask_shape_file_path).replace('.shp', '')

        # Define the output rasterized file path
        rasterized_file_path = os.path.join(rasterized_dir, f'{rasterized_file_name}_rasterized.tif')

        # Get projection info from reference image
        Image = gdal.Open(image_file_path, gdal.GA_ReadOnly)
        if Image is None:
            logger.error(f"Failed to open image file: {image_file_path}")
            return

        # Open Shapefile
        Shapefile = ogr.Open(mask_shape_file_path)
        if Shapefile is None:
            logger.error(f"Failed to open shapefile: {mask_shape_file_path}")
            return
        Shapefile_layer = Shapefile.GetLayer()

        # Rasterize
        logger.info(f"Rasterizing shapefile for {rasterized_file_name}")

        Output = gdal.GetDriverByName(gdalformat).Create(
            rasterized_file_path, Image.RasterXSize, Image.RasterYSize, 1,
            datatype, options=['COMPRESS=DEFLATE']
        )
        Output.SetProjection(Image.GetProjectionRef())
        Output.SetGeoTransform(Image.GetGeoTransform())

        # Write data to band 1
        Band = Output.GetRasterBand(1)
        Band.SetNoDataValue(0)

        gdal.RasterizeLayer(Output, [1], Shapefile_layer, options=['ATTRIBUTE=Class_id'])

        logger.info(f"Rasterization of shapefile for {rasterized_file_name} completed. Output saved to {rasterized_file_path}")

    except Exception as e:
        logger.exception(f"An error occurred while processing {rasterized_file_name}: {e}")

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


# def rasterize_masks(image_meta, intersecting_masks, image_height, image_width):
#     mask = np.zeros((1, image_height, image_width), dtype=np.uint8)
#
#     for idx, mask_geometry in intersecting_masks.iterrows():
#         mask_array = rasterize([mask_geometry.geometry],
#                                out_shape=(image_height, image_width))
#
#         mask |= mask_array.astype(np.uint8)
#
#     return mask

"""
def save_image_patch(patch, patch_path, image_meta):
    # Create directories if they don't exist
    os.makedirs(os.path.dirname(patch_path), exist_ok=True)

    # Save patch as TIFF image
    with rasterio.open(patch_path, 'w', **image_meta) as dst:
        dst.write(patch)


def save_mask_patch(patch, patch_path, mask_meta):
    # Create directories if they don't exist
    os.makedirs(os.path.dirname(patch_path), exist_ok=True)

    # Save patch as TIFF image
    with rasterio.open(patch_path, 'w', **mask_meta) as dst:
        dst.write(patch.astype(np.uint8)[np.newaxis, ...])
"""

def save_image_patch(patch, patch_path, image_meta):
    # Create directories if they don't exist
    os.makedirs(os.path.dirname(patch_path), exist_ok=True)

    # Save patch as TIFF image
    with rasterio.open(patch_path, 'w', **image_meta) as dst:
        dst.write(patch)


def save_mask_patch(patch, patch_path, mask_meta):
    # Create directories if they don't exist
    os.makedirs(os.path.dirname(patch_path), exist_ok=True)
    #  (after coorection)


    # Save patch as TIFF image
    with rasterio.open(patch_path, 'w', **mask_meta) as dst:
        dst.write(patch)

def prepare_dataset():
    image_path = './datasets/planet_dataset/composite.tif'
    mask_path = './datasets/custom_dataset/shape-files/Landcover_Clip.shp'
    output_dir = "datasets/custom_dataset/output/512x512"
    patch_size = 512
    stride = 256
    boundary = [3050511, 354090, 3066774, 375485]
    create_patches(image_path, mask_path, output_dir, patch_size, stride)


def test_data_generator():
    # Create the dataset
    data_path = "./datasets/custom_dataset/output/512x512/"
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    # transform = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.5], std=[0.5]),
    #     transforms.Lambda(lambda x: x * 255),
    # ])

    dataset = NepalDataset(data_path, transform=transform)

    # Create the data generator
    batch_size = 16
    shuffle = True
    data_generator = NepalDataGenerator(dataset, batch_size=batch_size, shuffle=shuffle)
    images, masks = data_generator.__next__()  # Get the batch of images and masks

    for i in range(0, 3):
        image = images[i].permute(1, 2, 0).numpy()  # Access individual image and convert to numpy array
        mask = masks[i].squeeze().numpy()  # Access individual mask and convert to numpy array
        visualize(image, mask)

def calculate_metrics(preds, targets, num_classes):
    """Calculate IoU, F1-score, and accuracy."""
    metrics = {"IoU": [], "F1-Score": [], "Pixel Accuracy": 0.0}

    # Convert predictions to class labels
    preds = torch.argmax(preds, dim=1)

    intersection = torch.zeros(num_classes, device=preds.device)
    union = torch.zeros(num_classes, device=preds.device)
    TP = torch.zeros(num_classes, device=preds.device)  # True Positives
    FP_FN_error = torch.zeros(num_classes, device=preds.device)  # False Positives + False Negatives

    for c in range(num_classes):
        pred_c = preds == c
        target_c = targets == c

        intersection[c] = torch.sum((pred_c & target_c).float())  # Intersection
        union[c] = torch.sum((pred_c | target_c).float())  # Union
        TP[c] = intersection[c]
        FP_FN_error[c] = torch.sum(pred_c.float()) + torch.sum(target_c.float()) - TP[c]

    # IoU per class
    metrics["IoU"] = (intersection / (union + 1e-7)).cpu().numpy()
    # F1-Score per class
    metrics["F1-Score"] = (2 * TP / (FP_FN_error + 2 * TP + 1e-7)).cpu().numpy()
    # Pixel Accuracy
    metrics["Pixel Accuracy"] = (torch.sum(intersection) / torch.sum(union)).item()

    return metrics

