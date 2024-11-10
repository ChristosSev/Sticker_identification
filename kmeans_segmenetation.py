
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from sklearn.cluster import KMeans
import cv2
import argparse

def segment_with_kmeans(img):
    pixels = img.reshape((-1, 1))
    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans.fit(pixels)
    labels = kmeans.labels_
    sorted_centers = np.sort(kmeans.cluster_centers_.flatten())
    threshold1 = (sorted_centers[0] + sorted_centers[1]) / 2
    threshold2 = sorted_centers[2]
    segmented_image = np.zeros_like(img)
    segmented_image[img < threshold1] = 0
    segmented_image[(img >= threshold1) & (img < threshold2)] = 127
    segmented_image[img >= threshold2] = 255
    return segmented_image

def colorize_segmented_image(segmented_image):
    colored_image = np.zeros((segmented_image.shape[0], segmented_image.shape[1], 3), dtype=np.uint8)
    colored_image[segmented_image == 0] = [0, 0, 255]
    colored_image[segmented_image == 127] = [255, 255, 0]
    colored_image[segmented_image == 255] = [255, 0, 0]
    return colored_image

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
    return largest_bounding_boxes

def main(image_path):
    img = io.imread(image_path)
    segmented_image = segment_with_kmeans(img)
    colored_segmented_image = colorize_segmented_image(segmented_image)
    bounding_boxes = identify_sticker_bounding_boxes(segmented_image)
    image_with_boxes = cv2.cvtColor(segmented_image, cv2.COLOR_GRAY2BGR)
    for box in bounding_boxes:
        cv2.drawContours(image_with_boxes, [box], 0, (0, 255, 0), 2)
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))
    ax[0, 0].imshow(img, cmap='gray')
    ax[0, 0].set_title('Original Image')
    ax[1, 0].imshow(segmented_image, cmap='gray')
    ax[1, 0].set_title('Segmented Image (Dark, Light, Gray)')
    ax[1, 1].imshow(colored_segmented_image)
    ax[1, 1].set_title('Colorized Segmentation')
    ax[0, 1].imshow(image_with_boxes)
    ax[0, 1].set_title('Segmented Image with Green Bounding Boxes')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Segment image using K-means and identify stickers with bounding boxes.")
    parser.add_argument('--image_path', type=str, default='/Users/christos/Downloads/dl_coding_challange24_25/0005_label.png',
                        help="Path to the input image")
    args = parser.parse_args()
    main(args.image_path)
