# frontend.py

import streamlit as st
import pandas as pd
from PIL import Image
# Make sure backendhybrid is accessible
from backend import filter_images # This will now use the globally loaded YOLO model implicitly

st.set_page_config(page_title="Eyeglasses Face Detector", layout="wide")
st.title("🧠 Eyeglasses Detection with Hybrid YOLO + CLIP")
st.markdown("""
This tool detects human faces **wearing eyeglasses** (not sunglasses) in images using a hybrid pipeline:
- ✅ Fast detection via YOLOv8
- ✅ Semantic fallback via CLIP
- ✅ CLAHE enhancement for black-and-white images
""")

# Button to trigger detection
if st.button("🔍 Run Detection"):
    with st.spinner("Processing images in parallel..."):
        # If you were using a parquet file, ensure it exists in the same directory or provide full path
        # For testing the provided images, the backend would need to be adapted to receive a list of paths directly
        # For now, this assumes a 'wit_with_images_5.parquet' file as per your original backend.
        # If you want to use the specific images you provided, you'd need to create a dummy parquet file
        # or modify the frontend to pass a list of image paths.
        results_df = filter_images(max_workers=4) # This calls the filter_images from backendhybrid

    if results_df.empty:
        st.warning("❌ No matching faces with eyeglasses found.")
    else:
        st.success(f"✅ Found {len(results_df)} matching face crops.")
        cols = st.columns(3)

        for i, row in results_df.iterrows():
            col = cols[i % 3]
            try:
                with col:
                    st.image(row["face_crop_path"], caption=row["caption"], width=200)
                    st.markdown(f"📘 **Page Title:** {row['page_title']}")
                    st.markdown(f"🔗 [Image URL]({row['image_url']})")
            except Exception as e:
                st.warning(f"⚠️ Could not load image: {row['face_crop_path']} ({e})")

else:
    st.info("👆 Click the button to run detection using the hybrid backend.")