import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from typing import List, Tuple, Dict, Optional


class SegmentationConfig:
    """Configuration class for segmentation parameters"""
    def __init__(self):
        # Line segmentation parameters
        self.line_gap_threshold_ratio = 0.1
        self.min_line_height = 8
        
        # Character filtering parameters
        self.min_width = 4
        self.min_height = 8
        self.min_aspect_ratio = 0.1
        self.max_aspect_ratio = 3.0
        self.min_height_ratio = 0.35
        self.max_height_ratio = 2.5
        self.min_absolute_area = 20
        
        # Overlap merging parameters
        self.overlap_threshold = 0.1
        
        # Morphological operation parameters
        self.morph_kernel_size = 3
        self.morph_iterations = 1


def merge_overlapping_boxes(boxes: List[Tuple[int, int, int, int]], 
                          overlap_thresh: float = 0.1) -> List[Tuple[int, int, int, int]]:
    """
    Merges overlapping or very close bounding boxes, which is crucial for
    correctly segmenting complex Devanagari characters.
    
    Args:
        boxes: List of bounding boxes as (x, y, w, h) tuples
        overlap_thresh: Threshold for merging overlapping boxes
        
    Returns:
        List of merged bounding boxes
    """
    if not boxes:
        return []
        
    boxes = sorted(boxes, key=lambda b: b[0])
    
    merged_boxes = []
    current_box = list(boxes[0])

    for next_box in boxes[1:]:
        x1, y1, w1, h1 = current_box
        x2, y2, w2, h2 = next_box
        
        # Check for horizontal proximity/overlap
        # If the start of the next box is within a certain threshold of the current box's end
        if x2 < (x1 + w1 + (min(w1, w2) * overlap_thresh)):
            new_x = min(x1, x2)
            new_y = min(y1, y2)
            new_w = max(x1 + w1, x2 + w2) - new_x
            new_h = max(y1 + h1, y2 + h2) - new_y
            current_box = [new_x, new_y, new_w, new_h]
        else:
            merged_boxes.append(tuple(current_box))
            current_box = list(next_box)
            
    merged_boxes.append(tuple(current_box))
    
    return merged_boxes


def load_images(original_path: str, binary_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load and validate input images.
    
    Args:
        original_path: Path to original image
        binary_path: Path to preprocessed binary image
        
    Returns:
        Tuple of (original_image, binary_image, gray_image)
        
    Raises:
        FileNotFoundError: If images cannot be loaded
        ValueError: If images have invalid dimensions
    """
    print("--- Step 1: Loading Images for Segmentation Pipeline ---")
    
    # Load original image
    original_image = cv2.imread(original_path)
    if original_image is None:
        raise FileNotFoundError(f"File not found at: {original_path}")
    print(f"  Original image '{original_path}' loaded.")

    # Load binary image
    binary_image = cv2.imread(binary_path, cv2.IMREAD_GRAYSCALE)
    if binary_image is None:
        raise FileNotFoundError(f"File not found at: {binary_path}")
    print(f"  Preprocessed binary image '{binary_path}' loaded.")

    # Convert to grayscale
    gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    print("  Original image converted to grayscale for high-quality cropping.")
    
    # Validate dimensions
    if original_image.shape[:2] != binary_image.shape:
        raise ValueError("Original and binary images must have the same dimensions")
    
    return original_image, binary_image, gray_image


def segment_lines(binary_image: np.ndarray, config: SegmentationConfig, 
                 show_plot: bool = True) -> List[Tuple[int, int]]:
    """
    Segment text into lines using horizontal projection profile.
    
    Args:
        binary_image: Binary image to segment
        config: Segmentation configuration
        show_plot: Whether to display the projection profile
        
    Returns:
        List of (start_y, end_y) line segments
    """
    print("\n--- Step 2: Line Segmentation ---")

    # Morphological opening to clean up noise and consolidate text blobs horizontally
    kernel = np.ones((config.morph_kernel_size, config.morph_kernel_size), np.uint8)
    opened_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel, 
                                  iterations=config.morph_iterations)
    
    # Create horizontal projection profile (sum of white pixels per row)
    horizontal_hist = np.sum(opened_image, axis=1) 
    
    if show_plot:
        plt.figure(figsize=(10, 6))
        plt.plot(horizontal_hist, range(len(horizontal_hist)))
        plt.gca().invert_yaxis() # Invert y-axis to match image coordinates
        plt.title('Horizontal Projection Profile (for Line Segmentation)')
        plt.xlabel('White Pixel Count (Brightness)')
        plt.ylabel('Row Number')
        plt.grid(True)
        plt.show(block=False) # Show non-blocking so the script can continue execution

    # Determine threshold for identifying gaps between lines
    line_gap_threshold = np.max(horizontal_hist) * config.line_gap_threshold_ratio
    print(f"  Line Detection Parameters: Threshold={line_gap_threshold:.1f}, Min Height={config.min_line_height}")

    # Find line segments by iterating through the horizontal profile
    in_text_block = False
    current_line_start = 0 
    line_segments = []
    
    for i, val in enumerate(horizontal_hist):
        if val > line_gap_threshold and not in_text_block:
            # Entering a text block
            in_text_block = True
            current_line_start = i 
        elif val <= line_gap_threshold and in_text_block:
            # Exiting a text block
            in_text_block = False
            end_y = i
            line_height = end_y - current_line_start
            if line_height > config.min_line_height:
                line_segments.append((current_line_start, end_y)) 
    
    # Handle the case where the last line extends to the end of the image
    if in_text_block:
        line_height = len(horizontal_hist) - current_line_start 
        if line_height >= config.min_line_height:
            line_segments.append((current_line_start, len(horizontal_hist))) 

    if not line_segments:
        print("  No valid lines found after line segmentation. Adjust 'line_gap_threshold' or 'min_line_height'.")
        if show_plot:
            plt.show() # Ensure plot is closed before returning
        return []

    print(f"  Successfully separated into {len(line_segments)} valid line(s).")
    if show_plot:
        print("  Close the 'Horizontal Projection Profile' plot to proceed.")
        plt.show() # Block until plot is closed
    
    return line_segments


def segment_characters_adaptively(original_image_path: str, binary_image_path: str, 
                                output_dir: str = "segmented_characters",
                                config: Optional[SegmentationConfig] = None,
                                show_plots: bool = True) -> Dict:
    """
    Performs a robust, adaptive character segmentation on a license plate image.
    This final version uses a two-stage filtering process to reject noise before
    calculating adaptive statistics.
    
    Args:
        original_image_path: Path to original image
        binary_image_path: Path to preprocessed binary image
        output_dir: Directory to save segmented characters
        config: Segmentation configuration (uses default if None)
        show_plots: Whether to display visualization plots
        
    Returns:
        Dictionary containing segmentation results and statistics
    """
    if config is None:
        config = SegmentationConfig()
    
    try:
        # Load images
        original_image, binary_image, gray_image = load_images(original_image_path, binary_image_path)
        
        # Segment lines
        line_segments = segment_lines(binary_image, config, show_plots)
        if not line_segments:
            return {"success": False, "error": "No valid lines found"}
        
        # Extract line segments from both binary and original grayscale images
        binary_line_segments = [binary_image[y1:y2, :] for y1, y2 in line_segments]
        original_gray_line_segments = [gray_image[y1:y2, :] for y1, y2 in line_segments]

        # Character segmentation
        print("\n--- Step 3: Performing Robust Character Segmentation with Merging ---")
        visualization_data = [] # To store data for plotting
        total_chars = 0

        for i, binary_line in enumerate(binary_line_segments):
            print(f"\n  Processing Line {i+1}...")
            # Find contours in the current binary line segment
            contours, _ = cv2.findContours(binary_line.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if not contours:
                print("    No initial contours found in this line.")
                # Store empty data for visualization
                visualization_data.append({
                    'original': original_gray_line_segments[i], 
                    'binary': binary_line, 
                    'all_contours': [], 
                    'valid_contours': [], 
                    'chars': []
                })
                continue

            # Stage 1: Absolute Pre-filtering - remove very small noise blobs
            stage1_contours = []
            for c in contours:
                x, y, w, h = cv2.boundingRect(c)
                if w >= config.min_width and h >= config.min_height:
                    stage1_contours.append(c)
            print(f"    Stage 1 (Pre-filter): Reduced {len(contours)} to {len(stage1_contours)} contours.")

            if not stage1_contours:
                print("    No valid contours remained after pre-filtering in this line.")
                visualization_data.append({
                    'original': original_gray_line_segments[i], 
                    'binary': binary_line, 
                    'all_contours': contours, # Keep original contours for visualization
                    'valid_contours': [], 
                    'chars': []
                })
                continue

            # Stage 2: Adaptive filtering based on median height, aspect ratio, and area
            all_heights = [cv2.boundingRect(c)[3] for c in stage1_contours]
            median_height = np.median(all_heights) if all_heights else 0
            print(f"    Stage 2 Reference Median Height: {median_height:.2f}")

            valid_contours = []
            for c in stage1_contours:
                x, y, w, h = cv2.boundingRect(c)
                area = cv2.contourArea(c)
                aspect_ratio = w / float(h) if h > 0 else 0

                # Height check: relative to median height
                height_check = (h > median_height * config.min_height_ratio) and \
                               (h < median_height * config.max_height_ratio)
                # Aspect ratio check
                aspect_check = (aspect_ratio > config.min_aspect_ratio) and \
                               (aspect_ratio < config.max_aspect_ratio)
                # Absolute area check
                area_check = area > config.min_absolute_area

                if height_check and aspect_check and area_check:
                    valid_contours.append(c)
            
            print(f"    Found {len(valid_contours)} valid character components before merging.")

            # Merge overlapping bounding boxes for multi-part characters
            valid_boxes = [cv2.boundingRect(c) for c in valid_contours]
            merged_boxes = merge_overlapping_boxes(valid_boxes, config.overlap_threshold)
            
            print(f"    Merged components into {len(merged_boxes)} final characters.")

            # Extract character images from the *original grayscale line segment*
            gray_line_to_crop = original_gray_line_segments[i]
            segmented_chars = [gray_line_to_crop[y:y+h, x:x+w] for x, y, w, h in merged_boxes]
            total_chars += len(segmented_chars)
            
            # Store data for visualization
            visualization_data.append({
                'original': gray_line_to_crop,
                'binary': binary_line,
                'all_contours': contours,
                'valid_contours': valid_contours,
                'chars': segmented_chars
            })

        # Visualization
        if show_plots:
            print("\n--- Step 4: Generating Visualization Plot ---")
            create_visualization(visualization_data)

        # Save characters
        print("\n--- Step 5: Saving Final Segmented Characters ---")
        saved_count = save_characters(visualization_data, output_dir)
        
        result = {
            "success": True,
            "total_lines": len(line_segments),
            "total_characters": total_chars,
            "saved_characters": saved_count,
            "output_directory": output_dir
        }
        
        print(f"\nScript finished successfully. Saved {saved_count} characters.")
        return result
        
    except Exception as e:
        print(f"\nERROR: {e}")
        return {"success": False, "error": str(e)}


def create_visualization(visualization_data: List[Dict]) -> None:
    """
    Create and display a multi-panel plot showing the segmentation steps and results.
    
    Args:
        visualization_data: List of dictionaries, each containing data for a line.
    """
    num_lines = len(visualization_data)
    # Determine the maximum number of characters in any line to set plot columns
    max_chars = max((len(d['chars']) for d in visualization_data), default=0)
    total_cols = 3 + max_chars # 3 for original, all_contours, valid_components + max_chars for segmented
    fig, axes = plt.subplots(num_lines, total_cols, figsize=(total_cols * 1.8, num_lines * 3.0), squeeze=False)
    fig.suptitle('Final Segmentation with Contour Merging', fontsize=16)

    for i, data in enumerate(visualization_data):
        # Original Line
        axes[i, 0].imshow(data['original'], cmap='gray')
        axes[i, 0].set_title(f'Line {i+1} Original')
        axes[i, 0].axis('off')

        # Binary Line with All Initial Contours
        line_with_all_contours = cv2.cvtColor(data['binary'], cv2.COLOR_GRAY2BGR)
        cv2.drawContours(line_with_all_contours, data['all_contours'], -1, (0, 255, 0), 1) # Green contours
        axes[i, 1].imshow(line_with_all_contours)
        axes[i, 1].set_title('All Contours')
        axes[i, 1].axis('off')

        # Binary Line with Valid (Filtered) Contours
        line_with_valid_contours = cv2.cvtColor(data['binary'], cv2.COLOR_GRAY2BGR)
        cv2.drawContours(line_with_valid_contours, data['valid_contours'], -1, (255, 0, 255), 1) # Magenta contours
        axes[i, 2].imshow(line_with_valid_contours)
        axes[i, 2].set_title('Valid Components')
        axes[i, 2].axis('off')

        # Segmented Characters
        for j, char_img in enumerate(data['chars']):
            if j + 3 < total_cols: # Ensure we don't go out of bounds for subplot
                axes[i, j + 3].imshow(char_img, cmap='gray')
                axes[i, j + 3].set_title(f'Char {j+1}', fontsize=8)
                axes[i, j + 3].axis('off')
        
        # Turn off any unused subplots in this row if lines have fewer characters than max_chars
        for j_clear in range(len(data['chars']) + 3, total_cols):
            axes[i, j_clear].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.95]) # Adjust layout to prevent title overlap
    print("  Displaying detailed plot. Close the plot window to save the characters.")
    plt.show() # This will block until the plot window is manually closed


def save_characters(visualization_data: List[Dict], output_dir: str) -> int:
    """
    Save segmented characters to individual image files in the specified directory.
    
    Args:
        visualization_data: List of dictionaries containing segmented character images.
        output_dir: Directory where character images will be saved.
        
    Returns:
        The total count of characters saved.
    """
    os.makedirs(output_dir, exist_ok=True) # Create output directory if it doesn't exist
    
    print(f"  Clearing old files from '{output_dir}'...")
    # Clear existing files in the output directory to ensure fresh results
    for f_name in os.listdir(output_dir):
        f_path = os.path.join(output_dir, f_name)
        try:
            if os.path.isfile(f_path): # Check if it's a file (not a subdirectory)
                os.remove(f_path)
        except Exception as e:
            print(f"    Could not remove file: {f_name}. Reason: {e}")

    # Check if there are any characters to save
    if not any(d['chars'] for d in visualization_data):
        print("  No valid characters were found to save.")
        return 0
        
    print("  Saving characters in a strict top-to-bottom, left-to-right sequence...")
    char_counter = 0
    total_saved_chars = 0
    
    # Iterate through lines and then characters within each line to save them sequentially
    for i, line_data in enumerate(visualization_data):
        for char_img in line_data['chars']:
            char_filename = f"char_{char_counter:02d}.png" # Format filename with leading zeros
            char_path = os.path.join(output_dir, char_filename)
            cv2.imwrite(char_path, char_img)
            char_counter += 1
        total_saved_chars += len(line_data['chars'])
    
    print(f"  Successfully saved a total of {total_saved_chars} characters to the '{output_dir}' folder.")
    return total_saved_chars


if __name__ == '__main__':
    # --- IMPORTANT: CHANGE THESE PATHS TO YOUR ACTUAL IMAGE FILES ---
    path_to_original_image = './images/AAA.png'
    path_to_preprocessed_binary_image = './final_binary_output.png'
    # ---------------------------------------------------------------
    
    # Instantiate the configuration with default values
    config = SegmentationConfig()
    
    # --- TUNE YOUR FILTERS HERE ---
    # Uncomment and change the values below to make the filters more or less strict.
    # Run the script after each change and observe the visualization plot.
    
    # -- To filter more small noise --
    config.min_width = 10          # Increase to ignore smaller-width blobs
    config.min_height = 10        # Increase to ignore smaller-height blobs
    
    # -- To adjust for character shapes --
    # config.min_aspect_ratio = 0.08 # Decrease if tall, thin parts are removed
    # config.max_aspect_ratio = 2  # Increase if wide parts (like top lines) are removed
    
    # -- To adjust for things small relative to the text line (like dots) --
    config.min_height_ratio = .9 # Decrease if valid dots/diacritics are removed
    
    # -- To adjust merging of character parts (like for 'i' or 'j') --
    
    
    # --- Run the main segmentation function ---
    result = segment_characters_adaptively(
        original_image_path=path_to_original_image,
        binary_image_path=path_to_preprocessed_binary_image,
        config=config,
        show_plots=True # Keep True to see the effect of your tuning
    )
    
    if not result["success"]:
        print(f"Segmentation failed: {result['error']}")
        sys.exit(1)

    print("\nSegmentation script finished its execution.")