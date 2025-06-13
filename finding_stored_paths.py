import pandas as pd
import os

# # Load your original CSV
df = pd.read_csv("wit_with_images_500.csv")

# # Define your image directory
# image_dir = "data"

# # Create new full path column for each image
# df["new_image_path"] = df["image_name"].apply(lambda name: os.path.join(image_dir, name))

# # Optional: verify the file exists before assigning
# df["new_image_path_exists"] = df["new_image_path"].apply(lambda x: os.path.exists(x))

# # Save updated CSV
# df.to_csv("wit_with_images_with_new_path.csv", index=False)

# print("✅ Updated CSV saved with 'new_image_path' column.")
df.to_parquet("wit_with_images_500.parquet", index=False)

print("✅ Saved to Parquet: wit_with_images_with_new_path.parquet")

