# Eyeglasses Detection with Hybrid YOLO + CLIP

This project implements a robust system for detecting human faces wearing eyeglasses (excluding sunglasses) in images. It utilizes a hybrid approach combining a Haar Cascade Classifier for initial face detection, YOLOv8 for eyeglasses detection, and OpenCLIP for semantic verification. The system is designed to handle various image qualities, including black-and-white and somewhat blurry images.

## Table of Contents

-   [Features](#features)
-   [Technical Stack](#technical-stack)
-   [Project Structure](#project-structure)
-   [Installation](#installation)
-   [Usage](#usage)
-   [Detection Process Overview](#detection-process-overview)
-   [Handling Challenging Images](#handling-challenging-images)
-   [Contributing](#contributing)
-   [License](#license)

## Features

* **Hybrid Detection Pipeline**: Combines Haar Cascades for face detection, YOLOv8 for primary eyeglasses/sunglasses detection, and CLIP for semantic verification.
* **Eyeglasses vs. Sunglasses Distinction**: Differentiates between eyeglasses and sunglasses using YOLOv8's class predictions.
* **Black & White Image Enhancement**: Automatically applies CLAHE (Contrast Limited Adaptive Histogram Equalization) to grayscale images for improved detection accuracy.
* **Face Cropping**: Crops detected faces and saves them with a slight padding for display and further processing (e.g., CLIP inference).
* **CSV-based Image Processing**: Reads image paths and metadata from a CSV file (`wit_with_images_500.csv`), processes them, and outputs results to a new CSV (`filtered_eyeglasses_detected.csv`).
* **Parallel Processing**: Leverages `ProcessPoolExecutor` for efficient, parallel image processing across multiple CPU cores.
* **Streamlit Frontend**: Provides an intuitive web interface for running the detection and viewing results.

## Technical Stack

The project is built primarily using Python and the following key libraries/frameworks:

* **Python 3.x**: The core programming language.
* **Streamlit**: For creating the interactive web application frontend.
* **OpenCV (`cv2`)**: Used for image processing tasks like reading images (`cv2.imread`), converting color spaces (`cv2.cvtColor`), applying CLAHE (`cv2.createCLAHE`, `clahe.apply`), and the Haar Cascade Classifier for face detection (`cv2.CascadeClassifier`).
* **`ultralytics` (YOLOv8)**: For object detection. Specifically, the `yolov8n.pt` (nano) model is used, globally loaded once for efficiency. It detects objects including `eyeglass` and `sunglass` classes based on its pre-trained weights.
* **`open_clip_torch`**: An open-source implementation of CLIP (Contrastive Language-Image Pre-training) based on `ViT-B-32` model pretrained on `laion2b_s34b_b79k` dataset. Used for semantic verification of eyeglasses presence when YOLO's confidence is low or it doesn't detect eyeglasses.
* **`torch` (PyTorch)**: The deep learning framework underlying YOLOv8 and CLIP.
* **`pandas`**: For efficient data handling, reading input CSV (`pd.read_csv`), and writing output CSV (`df.to_csv`).
* **`Pillow` (PIL)**: Used for image manipulation (e.g., `Image.open`, `Image.convert`, `Image.crop`, `Image.resize`).
* **`numpy`**: For numerical operations, especially with image data and CLIP feature vectors.
* **`scikit-learn`**: Specifically for `cosine_similarity` to compare CLIP image and text embeddings.
* **`concurrent.futures` (ProcessPoolExecutor)**: For parallelizing image processing tasks, improving throughput.

## Project Structure
├─── backend.py                       # The core logic for face and eyeglasses detection.
├─── frontend.py                      # Streamlit web application frontend.
├─── requirements.txt                 # List of Python dependencies.
├─── face_crops/                      # Directory to store cropped faces. (Created automatically)
├─── temp_enhanced_images/            # Temporary directory for CLAHE enhanced images for YOLO. (Created automatically)
├─── wit_with_images_500.csv          # Example input CSV file with image paths and metadata.
├─── filtered_eyeglasses_detected.csv # Output CSV file with detection results.
└─── Data/                            # Directory containing your actual image files (e.g., glassesgirl.jpg, etc.)

*Note: The primary backend logic is contained within `backend.py`, which is imported by `frontend.py`. Ensure that `backend.py` is in the same directory as `frontend.py` or accessible in Python's path.*

## Installation

Follow these steps to set up the project locally:

1.  **Clone the Repository (if applicable):**
    If you've already pushed your code to GitHub, clone it:
    ```bash
    git clone [https://github.com/GurubelliTanuj/ImageFinder.git](https://github.com/GurubelliTanuj/ImageFinder.git)
    cd ImageFinder
    ```
    If you haven't pushed yet, navigate to your project directory.

2.  **Verify Git Installation**:
    Make sure Git is installed and accessible in your terminal. If `git --version` doesn't work, install Git from [git-scm.com/download/win](https://git-scm.com/download/win) and restart your terminal.

3.  **Create a Virtual Environment (Recommended)**:
    ```bash
    python -m venv venv
    ```
    Activate the virtual environment:
    * **Windows**: `.\venv\Scripts\activate`
    * **macOS/Linux**: `source venv/bin/activate`

4.  **Install Dependencies**:
    The `requirements.txt` file lists all necessary Python packages.
    ```bash
    pip install -r requirements.txt
    ```

5.  **Download YOLOv8 Model**:
    The `backend.py` script uses `yolov8n.pt`. This model will be automatically downloaded by the `ultralytics` library the first time `YOLO("yolov8n.pt")` is called. No manual intervention is typically needed unless you want to ensure it's available offline.

6.  **Prepare your `wit_with_images_500.csv` file**:
    Ensure this CSV file exists in your project's root directory and contains an `image_path` column with correct (relative or absolute) paths to your images. Other columns like `caption_reference_description`, `page_title`, and `image_url` are optional but recommended for display in the frontend.

    Example `wit_with_images_500.csv` structure:
    ```csv
    image_path,caption_reference_description,page_title,image_url
    ./images/glassesgirl.jpg,A girl smiling with glasses,Happy Girl,file:///path/to/glassesgirl.jpg
    ./images/maleimagespecs.jpg,A man with round metal glasses,Male Portrait,file:///path/to/maleimagespecs.jpg
    # ... more image entries
    ```
    (Adjust `./images/` if your images are in a different subdirectory or directly in the project root).

## Usage

1.  **Run the Streamlit Application**:
    Ensure your virtual environment is active and you are in your project's root directory.
    ```bash
    streamlit run frontend.py
    ```
2.  **Interact with the Web UI**:
    Your browser will automatically open to the Streamlit application.
    Click the "🔍 Run Detection" button to start the image processing. The results, including face crops and metadata, will be displayed.

## Detection Process Overview

The `filter_images` function in `backend.py` is the entry point for the detection process. It orchestrates the analysis of each image from the input CSV in parallel.

1.  **Input Data Loading and Pre-processing (`filter_images` -> `process_row`):**
    * `filter_images` reads the input CSV (`wit_with_images_500.csv`) using `pd.read_csv`.
    * It then filters the DataFrame to include only valid `image_path` entries, checking for file existence.
    * Image processing is distributed across multiple workers using `ProcessPoolExecutor(max_workers=max_workers)`. Each row (representing an image) is passed to the `process_row` function.

2.  **Per-Image Processing (`process_row` function):**
    * **Image Loading & Grayscale Check**: The image specified by `row["image_path"]` is loaded using `cv2.imread`. `is_grayscale(image_path)` checks the mean RGB values (absolute difference less than 3 for R, G, B) to determine if the image is predominantly black and white.
    * **Conditional Image Enhancement (CLAHE)**: If `is_grayscale` returns `True`, `enhance_image_clahe(image_path)` is called. This function converts the image to LAB color space, applies `cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))` to the L-channel (luminance), and then converts back to BGR. The enhanced image is saved to a temporary file (`temp_enhanced_images/`) for YOLO processing.

3.  **Face Detection (`detect_face` function):**
    * A Haar Cascade Classifier (`haarcascade_frontalface_default.xml`) is loaded globally (`face_cascade_global`).
    * The input image (or its grayscale version) is converted to grayscale (`cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)`).
    * `face_cascade_global.detectMultiScale(gray, 1.1, 4)` is used to detect faces. The parameters `1.1` (scaleFactor) and `4` (minNeighbors) are used for multi-scale detection.
    * Detected faces are filtered based on minimum dimensions: `w >= MIN_FACE_WIDTH` (100 pixels) and `h >= MIN_FACE_HEIGHT` (100 pixels).
    * If no valid face is found, the image is skipped (`return None` from `process_row`).

4.  **Face Cropping and Resizing (`save_face_crop` function):**
    * If a face is detected, its bounding box (`x, y, w, h`) is used to crop the face region from the original image (opened via PIL `Image.open`).
    * A slight padding is added to the crop for better context.
    * The cropped face is resized to `(224, 224)` pixels, which is the expected input size for the CLIP model.
    * The processed face crop is saved to the `face_crops` directory, named after the original image file.

5.  **Eyeglasses/Sunglasses Detection (YOLOv8 - `detect_eyeglasses_yolo` function):**
    * The globally loaded YOLOv8 model (`yolo_model_global`) is used.
    * The `yolo_model_global(run_path, conf=CONFIDENCE_THRESHOLD)` performs inference. `run_path` will be the CLAHE-enhanced temporary image if it was grayscale, otherwise the original image path. `CONFIDENCE_THRESHOLD` is set to `0.25`.
    * Results (`results.boxes`) are parsed to extract predicted labels.
    * The function iterates through detected labels:
        * If `"sunglass"` is detected, it immediately returns `False` (not eyeglasses).
        * If `"eyeglass"` or `"glasses"` is detected, it returns `True`.
    * Temporary enhanced images are deleted after YOLO inference.

6.  **Semantic Verification (OpenCLIP - `clip_score` function - Fallback Mechanism):**
    * This step is only executed if YOLOv8 **did not** detect eyeglasses in the previous step.
    * The previously saved face crop (`crop_path`) is loaded and pre-processed for the CLIP model using `clip_preprocess`.
    * The `clip_model.encode_image` generates an image embedding for the face crop.
    * A text embedding for the phrase "a person wearing eyeglasses" is generated using `clip_model.encode_text` and `clip_tokenizer`.
    * Cosine similarity is calculated between the normalized image and text features using `cosine_similarity`.
    * If the calculated similarity (`sim`) is less than `CLIP_SIM_THRESHOLD` (set to `0.30`), the image is filtered out as not having eyeglasses. This provides a semantic confirmation or rejection.

7.  **Results Aggregation and Output:**
    * If an image successfully passes all face detection, YOLO detection (or CLIP verification), its details (original path, face crop path, caption, face box, page title, image URL) are collected.
    * All successful results are compiled into a pandas DataFrame, which is then saved to `filtered_eyeglasses_detected.csv`.

## Handling Challenging Images

* **Black and White Images**: The `is_grayscale` function identifies black and white images by checking the mean RGB values. For such images, `enhance_image_clahe` applies Contrast Limited Adaptive Histogram Equalization (CLAHE). CLAHE operates on the L-channel of the LAB color space, enhancing local contrast in different regions of the image, which helps both Haar Cascades and YOLOv8 discern features (like faces and eyeglasses) that might be obscured in poorly contrasted grayscale images.
* **Slightly Blurry Images**: While Haar Cascades are sensitive to blur, the CLAHE pre-processing can sometimes improve contrast enough to aid detection in images with moderate blur, especially if the blur primarily affects contrast. For severe blur, dedicated de-blurring techniques or more advanced face detection models (e.g., MTCNN, RetinaFace) would be required, which are beyond the scope of this project's current implementation.
* **Global Model Loading**: Crucially, the YOLOv8 model (`yolo_model_global`) and Haar Cascade classifier (`face_cascade_global`) are loaded only once at the start of the `backend.py` script. This prevents repeated model loading for every image processed by the `ProcessPoolExecutor`, drastically improving performance and making the process more efficient and robust against potential memory issues during large-scale batch processing.

## Contributing

Contributions are welcome! If you have suggestions for improvements, new features, or bug fixes, please feel free to open an issue or submit a pull request.

## License

This project is open-source and available under the [License Name or Link to License].
