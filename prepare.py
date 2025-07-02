import cv2
import os
import matplotlib.pyplot as plt

def create_binary_image(original_image_path: str, output_path: str):
    """
    Loads an image, converts it to grayscale, applies adaptive thresholding,
    and saves the resulting binary image. Includes visualization.

    Args:
        original_image_path (str): The path to the input image.
        output_path (str): The path where the binary image will be saved.
    """
    print("--- 🔬 Preprocessing Step ---")

    # Load the image in grayscale
    image = cv2.imread(original_image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Could not load image from: {original_image_path}")
    print(f"  ✅ Loaded '{original_image_path}'")

    # Apply adaptive thresholding. This inverts the image (THRESH_BINARY_INV)
    # so the text is white on a black background, as expected by cv2.findContours.
    binary_image = cv2.adaptiveThreshold(
        src=image,
        maxValue=255,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY_INV,
        blockSize=11,  # The size of the pixel neighborhood used to calculate the threshold.
        C=2            # A constant subtracted from the calculated mean.
    )
    print("  ✅ Applied adaptive thresholding to create binary image.")

    # Save the processed binary image
    cv2.imwrite(output_path, binary_image)
    print(f"  ✅ Saved binary image to '{output_path}'")

    # --- Visualization for the Preprocessing Step ---
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Original Grayscale Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(binary_image, cmap='gray')
    plt.title('Preprocessed Binary Image')
    plt.axis('off')

    plt.suptitle('Preprocessing Visualization', fontsize=16)
    print("  Displaying preprocessing result. Close the plot to continue.")
    plt.show()

    return output_path