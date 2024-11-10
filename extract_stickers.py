import numpy as np
import cv2
from skimage import io
import os
from typing import Tuple, List, Optional
import logging
from sklearn.cluster import KMeans
import shutil
import argparse

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class StickerDetectorBase:
    def __init__(self, min_area: int = 500, max_boxes: int = 2, padding: int = 10):
        self.min_area = min_area
        self.max_boxes = max_boxes
        self.padding = padding

    def extract_region(self, image: np.ndarray, box: np.ndarray) -> Optional[np.ndarray]:
        x, y, w, h = cv2.boundingRect(box)
        x_start = max(0, x - self.padding)
        y_start = max(0, y - self.padding)
        x_end = min(image.shape[1], x + w + self.padding)
        y_end = min(image.shape[0], y + h + self.padding)

        region = image[y_start:y_end, x_start:x_end]
        return region if region.size > 0 else None


class OtsuStickerDetector(StickerDetectorBase):
    def otsu_three_thresholds(self, image: np.ndarray) -> Tuple[np.ndarray, int, int]:
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        hist = cv2.calcHist([image], [0], None, [256], [0, 256]).flatten()
        prob = hist / image.size

        max_sigma_b = 0
        optimal_t1, optimal_t2 = 0, 0

        cumsum = np.cumsum(prob)
        mean_levels = np.arange(256)
        mean_cumsum = np.cumsum(mean_levels * prob)

        for t1 in range(1, 254):
            for t2 in range(t1 + 1, 255):
                w0 = cumsum[t1]
                w1 = cumsum[t2] - cumsum[t1]
                w2 = 1.0 - cumsum[t2]

                if min(w0, w1, w2) < 1e-10:
                    continue

                mu0 = mean_cumsum[t1] / w0 if w0 > 0 else 0
                mu1 = (mean_cumsum[t2] - mean_cumsum[t1]) / w1 if w1 > 0 else 0
                mu2 = (mean_cumsum[-1] - mean_cumsum[t2]) / w2 if w2 > 0 else 0

                mu_total = mean_cumsum[-1]

                sigma_b = (w0 * (mu0 - mu_total) ** 2 +
                           w1 * (mu1 - mu_total) ** 2 +
                           w2 * (mu2 - mu_total) ** 2)

                if sigma_b > max_sigma_b:
                    max_sigma_b = sigma_b
                    optimal_t1, optimal_t2 = t1, t2

        segmented = np.zeros_like(image)
        segmented[image <= optimal_t1] = 0
        segmented[(image > optimal_t1) & (image <= optimal_t2)] = 127
        segmented[image > optimal_t2] = 255

        return segmented, optimal_t1, optimal_t2

    def process_image(self, image_path: str, output_path: str) -> None:
        try:
            image = io.imread(image_path)
            segmented_image, _, _ = self.otsu_three_thresholds(image)
            bounding_boxes = self.identify_sticker_regions(segmented_image)

            if not bounding_boxes:
                logger.warning(f"No sticker regions found in {image_path} using Otsu method")
                return

            for i, box in enumerate(bounding_boxes, 1):
                region = self.extract_region(image, box)
                if region is not None:
                    output_file = os.path.join(
                        output_path,
                        f"{os.path.splitext(os.path.basename(image_path))[0]}_sticker_{i}.png"
                    )
                    cv2.imwrite(output_file, region)
                    logger.info(f"Saved sticker region {i} to {output_file} (Otsu)")

        except Exception as e:
            logger.error(f"Error processing {image_path} with Otsu method: {str(e)}")

    def identify_sticker_regions(self, segmented_image: np.ndarray) -> List[np.ndarray]:
        light_region = np.where(segmented_image == 255, 255, 0).astype(np.uint8)
        contours, _ = cv2.findContours(light_region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        valid_boxes = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area >= self.min_area:
                rect = cv2.minAreaRect(contour)
                box = cv2.boxPoints(rect)
                valid_boxes.append((area, np.int0(box)))

        valid_boxes.sort(key=lambda x: x[0], reverse=True)
        return [box for _, box in valid_boxes[:self.max_boxes]]


class KMeansStickerDetector(StickerDetectorBase):
    def segment_with_kmeans(self, image: np.ndarray) -> np.ndarray:
        pixels = image.reshape((-1, 1))
        kmeans = KMeans(n_clusters=3, random_state=42)
        kmeans.fit(pixels)

        sorted_centers = np.sort(kmeans.cluster_centers_.flatten())
        threshold1 = (sorted_centers[0] + sorted_centers[1]) / 2
        threshold2 = sorted_centers[2]

        segmented_image = np.zeros_like(image)
        segmented_image[image < threshold1] = 0
        segmented_image[(image >= threshold1) & (image < threshold2)] = 127
        segmented_image[image >= threshold2] = 255

        return segmented_image

    def identify_sticker_regions(self, segmented_image: np.ndarray) -> List[np.ndarray]:
        light_region = np.where(segmented_image == 255, 255, 0).astype(np.uint8)
        contours, _ = cv2.findContours(light_region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        valid_boxes = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area >= self.min_area:
                rect = cv2.minAreaRect(contour)
                box = cv2.boxPoints(rect)
                valid_boxes.append(np.int0(box))

        valid_boxes.sort(key=lambda x: cv2.contourArea(x), reverse=True)
        return valid_boxes[:self.max_boxes]

    def process_image(self, image_path: str, output_path: str) -> None:
        try:
            image = io.imread(image_path)
            segmented_image = self.segment_with_kmeans(image)
            bounding_boxes = self.identify_sticker_regions(segmented_image)

            if not bounding_boxes:
                logger.warning(f"No sticker regions found in {image_path} using K-means method")
                return

            for i, box in enumerate(bounding_boxes, 1):
                region = self.extract_region(image, box)
                if region is not None:
                    output_file = os.path.join(
                        output_path,
                        f"{os.path.splitext(os.path.basename(image_path))[0]}_sticker_{i}.png"
                    )
                    cv2.imwrite(output_file, region)
                    logger.info(f"Saved sticker region {i} to {output_file} (K-means)")

        except Exception as e:
            logger.error(f"Error processing {image_path} with K-means method: {str(e)}")


def organize_dataset(input_folder: str, output_folder: str, method: str) -> None:
    """
    Organize detected stickers into shipping and barcode folders.
    """
    shipping_folder = os.path.join(output_folder, f'shipping_{method}')
    barcode_folder = os.path.join(output_folder, f'barcode_{method}')

    os.makedirs(shipping_folder, exist_ok=True)
    os.makedirs(barcode_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if not any(filename.endswith(ext) for ext in ['.png', '.jpg', '.jpeg']):
            continue

        file_path = os.path.join(input_folder, filename)

        if 'sticker_1' in filename:
            shutil.copy(file_path, os.path.join(shipping_folder, filename))
            logger.info(f"Copied {filename} to shipping folder ({method})")
        elif 'sticker_2' in filename:
            shutil.copy(file_path, os.path.join(barcode_folder, filename))
            logger.info(f"Copied {filename} to barcode folder ({method})")


def process_pipeline(input_folder: str, output_folder: str, method: str, min_area: int, max_boxes: int,
                     padding: int) -> None:
    """
    Run the sticker detection and organization pipeline using the specified method.
    """
    # Create output directories
    detection_output = os.path.join(output_folder, f'detected_{method}')
    organized_output = os.path.join(output_folder, 'organized')

    os.makedirs(detection_output, exist_ok=True)
    os.makedirs(organized_output, exist_ok=True)

    # Initialize detector based on chosen method
    if method == 'otsu':
        detector = OtsuStickerDetector(min_area=min_area, max_boxes=max_boxes, padding=padding)
    else:  # kmeans
        detector = KMeansStickerDetector(min_area=min_area, max_boxes=max_boxes, padding=padding)

    # Process images
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(input_folder, filename)
            logger.info(f"Processing {filename} using {method} method")
            detector.process_image(image_path, detection_output)

    # Organize results
    organize_dataset(detection_output, organized_output, method)


def parse_arguments():
    parser = argparse.ArgumentParser(description='Sticker Detection Pipeline')

    parser.add_argument('--input', '-i',
                        required=True,
                        help='Input folder containing images')

    parser.add_argument('--output', '-o',
                        required=True,
                        help='Output folder for results')

    parser.add_argument('--method', '-m',
                        choices=['otsu', 'kmeans'],
                        required=True,
                        help='Detection method to use (otsu or kmeans)')

    parser.add_argument('--min-area',
                        type=int,
                        default=500,
                        help='Minimum area for region to be considered a sticker (default: 500)')

    parser.add_argument('--max-boxes',
                        type=int,
                        default=2,
                        help='Maximum number of sticker regions to detect (default: 2)')

    parser.add_argument('--padding',
                        type=int,
                        default=10,
                        help='Padding pixels around detected regions (default: 10)')

    return parser.parse_args()


if __name__ == "__main__":
    # Parse command line arguments
    args = parse_arguments()

    # Run the pipeline with specified arguments
    process_pipeline(
        input_folder=args.input,
        output_folder=args.output,
        method=args.method,
        min_area=args.min_area,
        max_boxes=args.max_boxes,
        padding=args.padding
    )
