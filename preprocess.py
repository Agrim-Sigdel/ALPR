import cv2
import numpy as np
import sys
import argparse

def order_points(pts):
    """
    Orders the four points of a rectangle in a consistent
    top-left, top-right, bottom-right, bottom-left sequence.
    """
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def deskew_plate(image):
    """
    Finds the license plate contour using robust shape analysis and applies a 
    perspective transform to get a deskewed, top-down view.
    """
    img_copy = image.copy()
    gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise and Canny to find edges
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 75, 200)
    
    # Find contours in the edged image
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # Sort contours by area in descending order and keep the top 10
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    
    plate_contour = None
    
    # Loop over the contours to find the best 4-point approximation
    for c in contours:
        perimeter = cv2.arcLength(c, True)
        # Approximate the contour to a polygon
        approx = cv2.approxPolyDP(c, 0.018 * perimeter, True)
        
        # If the approximated contour has 4 points, we assume it's the plate
        if len(approx) == 4:
            plate_contour = approx
            break

    if plate_contour is None:
        print("Warning: No suitable 4-point contour found for deskewing. Returning original image.")
        return image

    # Ensure the found contour has 4 points before reshaping
    ordered_box = order_points(plate_contour.reshape(4, 2))

    width_a = np.sqrt(((ordered_box[2][0] - ordered_box[3][0]) ** 2) + ((ordered_box[2][1] - ordered_box[3][1]) ** 2))
    width_b = np.sqrt(((ordered_box[1][0] - ordered_box[0][0]) ** 2) + ((ordered_box[1][1] - ordered_box[0][1]) ** 2))
    max_width = max(int(width_a), int(width_b))

    height_a = np.sqrt(((ordered_box[1][0] - ordered_box[2][0]) ** 2) + ((ordered_box[1][1] - ordered_box[2][1]) ** 2))
    height_b = np.sqrt(((ordered_box[0][0] - ordered_box[3][0]) ** 2) + ((ordered_box[0][1] - ordered_box[3][1]) ** 2))
    max_height = max(int(height_a), int(height_b))

    dst = np.array([
        [0, 0], [max_width - 1, 0],
        [max_width - 1, max_height - 1], [0, max_height - 1]], dtype="float32")

    transform_matrix = cv2.getPerspectiveTransform(ordered_box, dst)
    warped = cv2.warpPerspective(image, transform_matrix, (max_width, max_height))
    
    print("1. Plate deskewed successfully using shape analysis.")
    return warped

def fill_character_holes(binary_image):
    """
    Finds all contours in a binary image and fills them to produce solid
    character shapes, removing any internal black dots.
    """
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled_image = np.zeros_like(binary_image)
    cv2.drawContours(filled_image, contours, -1, (255), thickness=cv2.FILLED)
    return filled_image

def preprocess_image_for_ocr(image_input):
    """
    Applies a final, robust preprocessing pipeline to handle various image conditions.
    """
    # 1. Deskew the plate using robust shape analysis
    deskewed_image = deskew_plate(image_input)
    
    # 2. Convert to grayscale
    if len(deskewed_image.shape) == 3:
        gray_image = cv2.cvtColor(deskewed_image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = deskewed_image
    
    # 3. Denoise and Enhance
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    print("2. Image denoised with Gaussian Blur.")
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_image = clahe.apply(blurred_image)
    print("3. Contrast enhanced with CLAHE.")

    # 4. Binarize using Otsu's method
    _, binary_image = cv2.threshold(enhanced_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    print("4. Image binarized with Otsu's thresholding.")
    
    # 5. Fill character holes
    # Apply morphological closing to fill small gaps and holes
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    closed_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # Fill character holes using contour analysis
    solid_char_image = fill_character_holes(closed_image)
    
    # Apply additional dilation to make characters more solid
    dilation_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    solid_char_image = cv2.dilate(solid_char_image, dilation_kernel, iterations=1)
    
    print("5. Filled character holes and enhanced character solidity.")

    # 6. Standardize to white text on black background
    white_pixels = cv2.countNonZero(solid_char_image)
    total_pixels = solid_char_image.size
    
    if white_pixels > (total_pixels*.50):
        print("   -> Detected black-on-white image. Inverting.")
        standardized_binary_image = cv2.bitwise_not(binary_image)
    else:
        print("   -> Image is already white-on-black.")
        standardized_binary_image = binary_image

    return deskewed_image, standardized_binary_image

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess an image for OCR.')
    parser.add_argument('--input', required=False, default='A.jpg', help='Path to the input image file.')
    parser.add_argument('--output', default='final_binary_output.png', help='Path to save the processed binary image.')
    args = parser.parse_args()

    original_image = cv2.imread(args.input)
    if original_image is None:
        print(f"Error: Could not read image from path: {args.input}")
        sys.exit(1)
    print("Original image loaded successfully.\n")

    # Call the new robust preprocessing function
    deskewed_img, processed_img = preprocess_image_for_ocr(original_image)
    print("\nPreprocessing complete.")

    cv2.imwrite(args.output, processed_img)
    print(f"Final binary image saved as '{args.output}'")

    cv2.imshow('Original Image', original_image)
    cv2.imshow('1. Deskewed Plate', deskewed_img)
    cv2.imshow('2. Final Processed Image', processed_img)
    
    print("\nPress any key on an image window to exit.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
