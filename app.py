import streamlit as st
import numpy as np
import cv2
from PIL import Image
from sklearn.cluster import MiniBatchKMeans
import time

# --- Page Configuration ---
st.set_page_config(
    page_title="Image Segmentation Dashboard",
    page_icon="🎨",
    layout="wide"
)

# --- Header & Introduction ---
st.title("🎨 K-Means Image Segmentation")
st.subheader("Interactive Dashboard for Color-Based Segmentation")

with st.expander("🔬 Theory & Algorithm Steps"):
    st.write("""
    **Aim:** To segment images based on color similarity using the K-Means clustering algorithm.

    **K-Means Clustering:** An unsupervised machine learning algorithm that groups data into `k` clusters. For image segmentation, pixels are treated as data points.
    1. **Initialization:** `k` cluster centroids are randomly chosen.
    2. **Assignment:** Each pixel is assigned to the nearest centroid.
    3. **Update:** Centroids are updated to be the mean of all pixels in their cluster.
    4. **Repeat:** Steps 2-3 are repeated until the centroids no longer change significantly.
    
    In this application, `MiniBatchKMeans` is used for faster processing of large images.
    """)

st.markdown("---")

# --- Sidebar Controls ---
st.sidebar.header("⚙️ **Settings & Controls**")
uploaded_file = st.sidebar.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
k = st.sidebar.slider("Number of Clusters (k)", min_value=2, max_value=12, value=5, step=1)
mode = st.sidebar.radio("Segmentation Mode", ("BGR", "RGB", "Grayscale-style"))
st.sidebar.markdown("---")
st.sidebar.info("Developed for Computer Vision Lab | Symbiosis Institute of Technology")


# --- Main Logic ---
def perform_segmentation(img, mode, k):
    """Segments the image based on the selected mode and k."""
    if mode == "RGB":
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif mode == "Grayscale-style":
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Reshape the image to a 2D array of pixels
    if mode == "Grayscale-style":
        reshaped_img = img.reshape((-1, 1))
    else:
        reshaped_img = img.reshape((-1, 3))
    
    # Apply MiniBatchKMeans for faster computation
    kmeans = MiniBatchKMeans(n_clusters=k, n_init=10, random_state=42)
    kmeans.fit(reshaped_img)
    labels = kmeans.predict(reshaped_img)
    
    # Reconstruct the segmented image
    segmented_img = np.zeros_like(reshaped_img)
    for i in range(k):
        segmented_img[labels == i] = kmeans.cluster_centers_[i]
    
    # Reshape back to original dimensions
    segmented_img = segmented_img.reshape(img.shape).astype('uint8')
    
    return segmented_img

# --- Display Results ---
if uploaded_file is not None:
    # Read image
    image = Image.open(uploaded_file)
    original_image = np.array(image)
    original_image_bgr = cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR)

    # Use columns for side-by-side display
    col1, col2 = st.columns(2)

    with col1:
        st.header("🖼️ Original Image")
        st.image(original_image, caption="Original", use_container_width=True)

    with col2:
        st.header(f"🎨 Segmented Image")
        if st.button(f"Start Segmentation with k={k}"):
            start_time = time.time()
            # Use st.spinner to show progress
            with st.spinner(f"Segmenting in {mode} mode... Please wait..."):
                segmented_img = perform_segmentation(original_image_bgr, mode, k)
                
                # Convert segmented image to RGB for Streamlit display
                if mode == "Grayscale-style":
                    display_img = cv2.cvtColor(segmented_img, cv2.COLOR_GRAY2RGB)
                else:
                    display_img = cv2.cvtColor(segmented_img, cv2.COLOR_BGR2RGB)
            
            end_time = time.time()
            processing_time = round(end_time - start_time, 2)
            
            st.image(display_img, caption=f"Segmented with k={k}", use_container_width=True)
            st.write(f"⏱️ **Processing Time:** {processing_time} seconds")

            # Download button for the segmented image
            pil_segmented_img = Image.fromarray(display_img)
            buffer = cv2.imencode('.png', cv2.cvtColor(segmented_img, cv2.COLOR_BGR2RGB))[1].tobytes()
            st.download_button(
                label="Download Segmented Image",
                data=buffer,
                file_name=f"segmented_k{k}_{mode}.png",
                mime="image/png"
            )
else:
    st.info("Upload an image from the sidebar to begin!")
