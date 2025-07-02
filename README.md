

 Devanagari Automatic License Plate Recognition (ALPR)

[](https://opensource.org/licenses/MIT)
[](https://www.python.org/downloads/release/python-390/)

This repository contains a complete pipeline for an Automatic License Plate Recognition (ALPR) system specifically designed for license plates using the Devanagari script. The system leverages a YOLOv8 model for initial plate detection and a series of advanced OpenCV techniques for robust character segmentation.

 ⚙️ System Pipeline

The recognition process follows a sequential, two-stage pipeline:

1.  Plate Detection & Cropping (`main.py`):

      * A YOLOv8 object detection model scans the input image to locate the vehicle's license plate.
      * The detected plate is automatically cropped and saved as a separate image file for the next stage.

2.  Character Segmentation (`segment.py`):

      * The cropped plate image undergoes extensive preprocessing, including deskewing and contrast enhancement.
      * The script identifies individual lines of text on the plate.
      * A specialized algorithm removes the shirorekha (the horizontal connecting line in Devanagari script) to separate joined letters.
      * Each segmented character is saved as an individual image, ready for input into an Optical Character Recognition (OCR) engine.

 ✨ Features

  * High-Accuracy Detection: Utilizes a state-of-the-art YOLOv8 model for reliable license plate detection from full images.
  * Advanced Image Processing: Implements a robust pipeline including perspective transformation and adaptive thresholding to handle various lighting conditions.
  * Specialized Devanagari Handling: Features a custom *shirorekha* removal algorithm, which enables accurate segmentation of connected Devanagari characters—a common failure point for standard ALPR systems.
  * Modular Two-Stage Process: The detection and segmentation steps are separated, allowing for independent testing and improvement.
  * Debugging Visualizations: The segmentation script can generate detailed plots showing the output of each stage, simplifying tuning and debugging.

 🚀 Getting Started

Follow these steps to set up and run the project locally.

# Prerequisites

  * Python 3.9 or higher
  * `pip` package manager

# Installation

1.  Clone the repository:

    ```bash
    git clone https://github.com/Agrim-Sigdel/ALPR.git
    cd ALPR
    ```

2.  Create and activate a virtual environment (recommended):

      * macOS / Linux:
        ```bash
        python3 -m venv venv
        source venv/bin/activate
        ```
      * Windows:
        ```bash
        python -m venv venv
        venv\Scripts\activate
        ```

3.  Install dependencies from `requirements.txt`:

    ```bash
    pip install -r requirements.txt
    ```

4.  Download YOLOv8 Model:
    Place your trained YOLOv8 license plate detection model (e.g., `best.pt`) in the root directory or provide a path to it when running the script.

 Usage

The pipeline is run in two sequential commands.

# Step 1: Detect and Crop the License Plate

Run `main.py` to detect the plate from a source image. The cropped plate will be saved in the `runs/detect/predict/crops/license_plate` directory.

```bash
python main.py --weights path/to/your/best.pt --source path/to/your/image.jpg
```

# Step 2: Segment Characters from the Cropped Plate

Run `segment.py` using the cropped image from the previous step as input.

```bash
python segment.py --original_image_path path/to/cropped_plate.jpg --binary_image_path path/to/binary_image.png
```

*Note: The `segment.py` script currently requires two separate paths for the original and a pre-processed binary image. You may need to generate the binary image first or modify the script to handle it internally.*

 📂 Project Structure

```
.
├── models/                 # (Optional) For storing YOLO models
├── runs/                   # Default output directory for YOLO detection
│   └── detect/predict/
│       └── crops/
│           └── license_plate/  # Cropped plates are saved here
├── segmented_characters/   # Default output for segmented characters
├── main.py                 # Script for plate detection (YOLOv8)
├── segment.py              # Script for character segmentation
├── requirements.txt        # Project dependencies
└── README.md
```

 📄 License

This project is licensed under the MIT License. See the `LICENSE` file for details.
