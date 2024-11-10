import argparse
import matplotlib.pyplot as plt
from skimage import io
import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Image segmentation using Otsu's multi-threshold method.")
    parser.add_argument(
        '--input_image',
        type=str,
        default='/Users/christos/Downloads/dl_coding_challange24_25/0001_label.png',
        help="Path to the input image"
    )
    return parser.parse_args()


def otsu_three_thresholds(image):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    hist, bin_edges = np.histogram(image, bins=256, range=(0, 256))
    hist = hist.astype(float)
    total_pixels = image.size
    prob = hist / total_pixels

    max_sigma_b = 0
    optimal_t1, optimal_t2 = 0, 0

    for t1 in range(1, 255):
        for t2 in range(t1 + 1, 256):
            w0 = np.sum(prob[:t1])
            w1 = np.sum(prob[t1:t2])
            w2 = np.sum(prob[t2:])

            if w0 == 0 or w1 == 0 or w2 == 0:
                continue

            mu0 = np.sum(np.arange(0, t1) * prob[:t1]) / w0
            mu1 = np.sum(np.arange(t1, t2) * prob[t1:t2]) / w1
            mu2 = np.sum(np.arange(t2, 256) * prob[t2:]) / w2

            mu_total = w0 * mu0 + w1 * mu1 + w2 * mu2
            sigma_b = w0 * (mu0 - mu_total) ** 2 + w1 * (mu1 - mu_total) ** 2 + w2 * (mu2 - mu_total) ** 2

            if sigma_b > max_sigma_b:
                max_sigma_b = sigma_b
                optimal_t1, optimal_t2 = t1, t2

    segmented_image = np.zeros_like(image)
    segmented_image[image <= optimal_t1] = 0
    segmented_image[(image > optimal_t1) & (image <= optimal_t2)] = 127
    segmented_image[image > optimal_t2] = 255

    return segmented_image, optimal_t1, optimal_t2


def identify_sticker_bounding_boxes(segmented_image, min_area=500, max_boxes=2):
    light_region = np.where(segmented_image == 255, 255, 0).astype(np.uint8)
    contours, _ = cv2.findContours(light_region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bounding_boxes = []
    for contour in contours:
        if cv2.contourArea(contour) < min_area:
            continue
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        bounding_boxes.append((cv2.contourArea(contour), box))

    bounding_boxes = sorted(bounding_boxes, key=lambda x: x[0], reverse=True)
    largest_bounding_boxes = [box for _, box in bounding_boxes[:max_boxes]]
    print(len(largest_bounding_boxes))

    return largest_bounding_boxes


def colorize_segmented_image(segmented_image):
    colored_image = np.zeros((segmented_image.shape[0], segmented_image.shape[1], 3), dtype=np.uint8)
    colored_image[segmented_image == 0] = [255, 0, 0]
    colored_image[segmented_image == 127] = [255, 255, 0]
    colored_image[segmented_image == 255] = [0, 0, 255]
    return colored_image


if __name__ == "__main__":
    args = parse_args()
    image = io.imread(args.input_image)
    segmented_image, T1, T2 = otsu_three_thresholds(image)

    colored_segmented_image = colorize_segmented_image(segmented_image)
    bounding_boxes = identify_sticker_bounding_boxes(segmented_image)

    image_with_boxes = cv2.cvtColor(segmented_image, cv2.COLOR_GRAY2BGR)
    for box in bounding_boxes:
        cv2.drawContours(image_with_boxes, [box], 0, (0, 255, 0), 2)

    print(f"Optimal thresholds: T1 = {T1}, T2 = {T2}")

    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))

    ax[0, 0].imshow(image, cmap='gray')
    ax[0, 0].set_title('Original Image')

    ax[1, 0].imshow(segmented_image, cmap='gray')
    ax[1, 0].set_title('Segmented Image')

    ax[1, 1].imshow(colored_segmented_image)
    ax[1, 1].set_title('Colored Segmented Image')

    ax[0, 1].imshow(image_with_boxes)
    ax[0, 1].set_title('Segmented Image with Bounding Boxes')

    plt.tight_layout()
    plt.show()
