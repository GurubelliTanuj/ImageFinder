from datasets import load_dataset
import pandas as pd
import os
import requests
from PIL import Image
from io import BytesIO
import re

def sanitize_filename(name):
    if not name:
        return "untitled"
    name = str(name)
    name = re.sub(r'[\\/*?:"<>|]', "_", name)
    return name.strip().replace(" ", "_")

# Setup
output_folder = "data/"
os.makedirs(output_folder, exist_ok=True)
ds = load_dataset("wikimedia/wit_base", split="train", streaming=True)

# Initialize empty DataFrame
df = pd.DataFrame(columns=[
    "image_name", "image_path", "image_url", 
    "original_height", "original_width", 
    "caption_reference_description", "page_title"
])

success_count = 0
target_count = 200  # ✅ Change this to the number of images you want

for i, row in enumerate(ds):
    if success_count >= target_count:
        break

    try:
        wit = row.get("wit_features", {})
        page_title_raw = wit.get("page_title")[0] if wit.get("page_title") else f"no_title_{i}"
        page_title_clean = sanitize_filename(page_title_raw)

        image_url = row.get("image_url")
        if not image_url:
            raise ValueError("Missing image URL")

        response = requests.get(image_url, timeout=5)
        content_type = response.headers.get("Content-Type", "")
        if not content_type.startswith("image"):
            raise ValueError(f"Invalid content-type: {content_type}")

        img = Image.open(BytesIO(response.content)).convert("RGB")
        width, height = img.size

        image_name = f"image_{page_title_clean}.jpg"
        image_path = os.path.join(output_folder, image_name)
        img.save(image_path)

        # ✅ Add metadata row directly to DataFrame
        df.loc[len(df)] = {
            "image_name": image_name,
            "image_path": image_path,
            "image_url": image_url,
            "original_height": height,
            "original_width": width,         
            "caption_reference_description": row.get("caption_reference_description"),
            "page_title": page_title_raw
        }

        success_count += 1
        print(f"[{i}] ✅ Downloaded and metadata added for: {image_name}")

    except Exception as e:
        print(f"[{i}] ❌ Failed: {e}")
        continue

# Save metadata
df.to_csv("wit_with_images_500.csv", index=False)
print("✅ Final metadata saved to wit_with_images_500.csv")
print(df.head())
