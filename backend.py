# backendhybrid.py

import os
import cv2
import pandas as pd
from PIL import Image, ImageStat
from concurrent.futures import ProcessPoolExecutor
from ultralytics import YOLO
import torch
import open_clip
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Constants
MIN_FACE_WIDTH = 100
MIN_FACE_HEIGHT = 100
CROP_FOLDER = "face_crops"
# Changed to CSV_PATH as requested previously
CSV_PATH = "wit_with_images_500.csv" # Default CSV file name
CLIP_SIM_THRESHOLD = 0.30
CONFIDENCE_THRESHOLD = 0.25 # YOLO confidence threshold

os.makedirs(CROP_FOLDER, exist_ok=True)
os.makedirs("temp_enhanced_images", exist_ok=True) # Ensure temp dir exists for enhanced images

# Load CLIP (once globally)
clip_model, _, clip_preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
clip_model = clip_model.to("cpu").eval()  # Set to CPU for light systems
clip_tokenizer = open_clip.get_tokenizer('ViT-B-32')

# Load YOLO (once globally) - IMPORTANT FIX
yolo_model_global = YOLO("yolov8n.pt")

# Load Haar cascade classifier (once globally)
face_cascade_global = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


def is_grayscale(image_path):
    """
    Checks if an image is predominantly grayscale.
    """
    try:
        img = Image.open(image_path).convert("RGB")
        stat = ImageStat.Stat(img)
        r, g, b = stat.mean
        # Check if R, G, B values are very close
        return abs(r - g) < 3 and abs(r - b) < 3
    except Exception as e:
        print(f"Error checking grayscale for {image_path}: {e}")
        return False # Assume not grayscale if error


def enhance_image_clahe(image_path):
    """
    Applies CLAHE (Contrast Limited Adaptive Histogram Equalization) to an image.
    Returns the enhanced image as a NumPy array.
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image for CLAHE enhancement {image_path}")
        return None

    # Convert to LAB color space
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab) # Split into L, A, B channels

    # Apply CLAHE to the L-channel (lightness)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)

    # Merge the enhanced L-channel with original A and B channels
    enhanced_lab = cv2.merge((cl, a, b))
    # Convert back to BGR color space
    return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)


def detect_face(image_path):
    """
    Detects faces in an image using Haar Cascade Classifier.
    Applies CLAHE if the image is grayscale for better detection.
    """
    img_to_process = None
    is_bw = is_grayscale(image_path) # Check if it's BW

    if is_bw:
        enhanced_img = enhance_image_clahe(image_path)
        if enhanced_img is not None:
            img_to_process = enhanced_img
        else:
            # Fallback to original if enhancement fails
            img_to_process = cv2.imread(image_path)
    else:
        img_to_process = cv2.imread(image_path)

    if img_to_process is None:
        print(f"Error: Could not read image for face detection {image_path}")
        return False, None

    gray = cv2.cvtColor(img_to_process, cv2.COLOR_BGR2GRAY)
    # Applying a slight blur can sometimes help with noisy images for Haar cascades
    # gray = cv2.medianBlur(gray, 3) # Optional: uncomment to add a slight median blur

    faces = face_cascade_global.detectMultiScale(gray, 1.1, 4, minSize=(MIN_FACE_WIDTH, MIN_FACE_HEIGHT))

    # The minSize parameter ensures only faces >= 100x100 are detected
    if len(faces) > 0:
        # Return the first detected face that meets the size criteria
        return True, faces[0] # Assuming we only care about the first large enough face
    return False, None


def save_face_crop(image_path, box):
    """
    Crops the face from the original image and saves it.
    """
    try:
        img = Image.open(image_path).convert("RGB")
        x, y, w, h = box
        # Add a small padding to the crop for better context, ensuring it stays within image bounds
        pad_x = int(w * 0.1)
        pad_y = int(h * 0.1)
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(img.width, x + w + pad_x)
        y2 = min(img.height, y + h + pad_y)

        crop = img.crop((x1, y1, x2, y2))
        crop = crop.resize((224, 224)) # Resize for CLIP input
        
        crop_filename = os.path.basename(image_path)
        crop_path = os.path.join(CROP_FOLDER, crop_filename)
        crop.save(crop_path)
        return crop_path
    except Exception as e:
        print(f"Error saving face crop for {image_path}: {e}")
        return None


def detect_eyeglasses_yolo(yolo_model, image_path, clahe=False):
    """
    Detects eyeglasses (and filters sunglasses) using YOLOv8.
    Optionally applies CLAHE enhancement.
    """
    try:
        run_path = image_path
        # If CLAHE is requested, save enhanced image to a temporary file for YOLO
        if clahe:
            enhanced_img_np = enhance_image_clahe(image_path)
            if enhanced_img_np is not None:
                temp_filename = f"temp_enhanced_{os.path.basename(image_path)}"
                run_path = os.path.join("temp_enhanced_images", temp_filename)
                cv2.imwrite(run_path, enhanced_img_np)
            else:
                # If enhancement failed, proceed with original image
                print(f"CLAHE enhancement failed for {image_path}, using original image for YOLO.")
                run_path = image_path # Revert to original path

        results = yolo_model(run_path, conf=CONFIDENCE_THRESHOLD)[0]
        labels = [yolo_model.names[int(box.cls[0])].lower() for box in results.boxes]

        # Clean up temporary file if it was created
        if run_path != image_path and os.path.exists(run_path):
            os.remove(run_path)

        for lbl in labels:
            if "sunglass" in lbl or "shades" in lbl: # Added "shades" as well
                return False
            if "eyeglass" in lbl or "glasses" in lbl:
                return True
        return False
    except Exception as e:
        print(f"[YOLO error] {image_path}: {e}")
        return False


def clip_score(image_path):
    """
    Calculates CLIP similarity score for "a person wearing eyeglasses".
    """
    try:
        img = Image.open(image_path).convert("RGB")
        image_tensor = clip_preprocess(img).unsqueeze(0).to("cpu")

        with torch.no_grad():
            image_features = clip_model.encode_image(image_tensor)
            text_features = clip_model.encode_text(clip_tokenizer(["a person wearing eyeglasses"]).to("cpu"))

        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        sim = cosine_similarity(image_features.cpu(), text_features.cpu())[0][0]
        return float(sim)
    except Exception as e:
        print(f"[CLIP error] {image_path}: {e}")
        return 0.0


def process_row(row):
    """
    Processes a single image row for face and eyeglasses detection.
    """
    image_path = row["image_path"]

    # This check is now redundant if filter_images already filtered, but harmless
    if not os.path.exists(image_path):
        print(f"File not found during processing (skipped by initial check): {image_path}")
        return None

    try:
        is_bw = is_grayscale(image_path)
        has_face, face_box = detect_face(image_path)
        if not has_face:
            # print(f"No valid face detected in {image_path}") # Can be noisy
            return None

        # Use the globally loaded YOLO model
        has_specs = detect_eyeglasses_yolo(yolo_model_global, image_path, clahe=is_bw)
        
        # Only save crop if a face is detected and we potentially need it for CLIP
        if has_specs or not has_specs: # Always save crop to check CLIP if YOLO fails
            crop_path = save_face_crop(image_path, face_box)
            if crop_path is None: # Handle case where crop saving failed
                print(f"Failed to save face crop for {image_path}")
                return None
        else:
            crop_path = None


        if not has_specs: # If YOLO didn't find specs, try CLIP
            if crop_path: # Only run CLIP if a crop was successfully saved
                score = clip_score(crop_path)
                if score < CLIP_SIM_THRESHOLD:
                    # print(f"No eyeglasses detected by YOLO and CLIP score too low ({score:.2f}) for {image_path}") # Can be noisy
                    return None
            else: # If no crop, cannot run CLIP
                return None
        
        # If we reach here, either YOLO found specs or CLIP confirmed it
        return {
            "image_path": image_path,
            "face_crop_path": crop_path,
            "caption": row.get("caption_reference_description", "No caption provided") or row.get("page_title", "No caption"),
            "face_box": str(face_box),
            "page_title": row.get("page_title", ""),
            "image_url": row.get("image_url", "")
        }
    except Exception as e:
        print(f"❌ Failed for {image_path}: {e}")
        return None


def filter_images(csv_path=CSV_PATH, max_workers=2):
    """
    Reads a CSV file, verifies image paths, and then processes images
    to detect faces with eyeglasses.
    """
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at {csv_path}. Please ensure it exists.")
        return pd.DataFrame()

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading CSV file {csv_path}: {e}")
        return pd.DataFrame()

    original_row_count = len(df)
    print(f"🚀 Loaded {original_row_count} entries from {csv_path}.")

    # --- Static File Path Check ---
    print("\n--- Verifying Image Paths in CSV File ---")
    valid_image_data = []
    invalid_paths_count = 0

    if 'image_path' not in df.columns:
        print("Error: 'image_path' column not found in the CSV file. Please check column names.")
        return pd.DataFrame()

    for index, row in df.iterrows():
        image_path = row.get("image_path")
        if pd.isna(image_path):
            invalid_paths_count += 1
            continue

        image_path = str(image_path).strip()

        if image_path and os.path.exists(image_path):
            valid_image_data.append(row.to_dict())
        else:
            invalid_paths_count += 1
            # print(f"❌ Missing or invalid image path: '{image_path}' (from row index {index})")

    print(f"✅ Found {len(valid_image_data)} valid image paths.")
    if invalid_paths_count > 0:
        print(f"⚠️ Skipped {invalid_paths_count} entries due to missing, invalid, or empty image paths.")
    # --- End of Static File Path Check ---

    if not valid_image_data:
        print("No valid image paths found after verification. Returning empty DataFrame.")
        return pd.DataFrame()

    print(f"\n🚀 Processing {len(valid_image_data)} valid images using {max_workers} workers...")

    # Using ThreadPoolExecutor as `process_row` involves significant I/O (image loading)
    # and CPU-bound (model inference). For optimal performance on multi-core systems,
    # ProcessPoolExecutor is generally better for CPU-bound tasks.
    # However, if your environment has constraints or if I/O is truly dominant, ThreadPoolExecutor might be considered.
    # Sticking with ProcessPoolExecutor for model inference.
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(process_row, valid_image_data)) # Pass list of dicts

    results = [r for r in results if r is not None]
    df_filtered = pd.DataFrame(results)
    df_filtered.to_csv("filtered_eyeglasses_detected.csv", index=False) # Changed output filename
    print(f"✅ Done: {len(df_filtered)} matches saved to filtered_eyeglasses_detected.csv.")
    return df_filtered