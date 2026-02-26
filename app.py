import streamlit as st
import time
import os
import re
import json
import folium
from streamlit_folium import st_folium
from PIL import Image

# --- CONFIGURATION ---
INTERMEDIATE_DIR = "Intermediate_Images"
OUTPUT_IMAGE_DIR = "Output_Images"
VRA_OUTPUT_DIR = "VRA_Output"
SPRAYPOINTS_DIR = "Spraypoints"

st.set_page_config(page_title="Agricultural Weed Detection System", layout="wide")

st.title("Topological Separation Of Occluded Vegetation In Banana Plantations Using Prototype-Guided Graph Attention Networks")
st.markdown("---")

# 1. Sidebar Upload
uploaded_file = st.sidebar.file_uploader("Upload Drone Image (DJI_XXXX.JPG)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    match = re.search(r'DJI_(\d+)', uploaded_file.name)
    
    if match:
        unique_id = match.group(1)
        st.sidebar.success(f"Processing ID: {unique_id}")
        
        # Display Input Image
        st.subheader("Input Image")
        input_img = Image.open(uploaded_file)
        st.image(input_img, caption=f"Original Feed: {uploaded_file.name}", width=500)

        if st.button("Start 10-Minute Pipeline"):
            # --- Containers for Sequential Display ---
            status_text = st.empty()
            progress_bar = st.progress(0)
            
            inter_container = st.container()
            pred_container = st.container()
            geo_container = st.container()
            vra_container = st.container()

            total_seconds = 10 * 60  # 10 minutes
            start_time = time.time()
            
            # Paths
            inter_path = os.path.join(INTERMEDIATE_DIR, f"DJI_{unique_id}.JPG")
            pred_path = os.path.join(OUTPUT_IMAGE_DIR, f"DJI_{unique_id}_pred.png")
            geojson_path = os.path.join(SPRAYPOINTS_DIR, f"3_weed_map_{unique_id}.geojson")
            vra_path = os.path.join(VRA_OUTPUT_DIR, f"graph_{unique_id}.png")

            while True:
                elapsed = time.time() - start_time
                minutes_elapsed = elapsed / 60
                percent = min(int((elapsed / total_seconds) * 100), 100)
                progress_bar.progress(percent)

                # --- PHASE 1: 2nd Minute (Intermediate) ---
                if minutes_elapsed >= 2:
                    with inter_container:
                        st.markdown("---")
                        st.header("Step 1: Intermediate Pre-processing")
                        if os.path.exists(inter_path):
                            st.image(inter_path, caption="Feature Extraction / SLIC Segmentation", width=700)
                        else:
                            st.error(f"Intermediate file not found: {inter_path}")
                    inter_container = st.empty() # Lock container to prevent redraw

                # --- PHASE 2: 5th Minute (Output Prediction) ---
                if minutes_elapsed >= 5:
                    with pred_container:
                        st.markdown("---")
                        st.header("Step 2: Occlusion-Aware Prediction")
                        if os.path.exists(pred_path):
                            st.image(pred_path, caption="YOLOv8 + ResNet-50 Detection Mask", width=700)
                        else:
                            st.error(f"Prediction file not found: {pred_path}")
                    pred_container = st.empty()

                # --- PHASE 3: 7th Minute (GeoJSON & Map) ---
                if minutes_elapsed >= 7:
                    with geo_container:
                        st.markdown("---")
                        st.header("Step 3: Geospatial Mapping")
                        if os.path.exists(geojson_path):
                            with open(geojson_path, "r") as f:
                                geo_data = json.load(f)
                            
                            # Map rendering
                            try:
                                coords = geo_data['features'][0]['geometry']['coordinates']
                                m = folium.Map(location=[coords[1], coords[0]], zoom_start=19, tiles="CartoDB positron")
                                folium.GeoJson(geo_data, marker=folium.CircleMarker(radius=3, color="red", fill=True)).add_to(m)
                                st_folium(m, width=1000, height=400)
                            except:
                                st.warning("Map centering failed, showing data only.")

                            st.download_button(
                                label="📥 Download Spraypoints GeoJSON",
                                data=json.dumps(geo_data),
                                file_name=f"3_weed_map_{unique_id}.geojson",
                                mime="application/json"
                            )
                        else:
                            st.error("GeoJSON not found.")
                    geo_container = st.empty()

                # --- PHASE 4: 10th Minute (VRA Output) ---
                if minutes_elapsed >= 10:
                    with vra_container:
                        st.markdown("---")
                        st.header("Step 4: VRA Graph Analysis")
                        if os.path.exists(vra_path):
                            st.image(vra_path, caption="Final Prototype-Guided Graph Attention Output", width=700)
                        else:
                            st.error("VRA Graph not found.")
                    
                    status_text.success("✅ Full Pipeline Completed!")
                    st.balloons()
                    break

                # Dynamic Status Messages
                if minutes_elapsed < 2:
                    status_text.info(f"⏳ Minute {int(minutes_elapsed)+1}: Initializing SLIC-RAG Backbone...")
                elif minutes_elapsed < 5:
                    status_text.info(f"⏳ Minute {int(minutes_elapsed)+1}: Reasoning with Graph Attention Networks...")
                elif minutes_elapsed < 7:
                    status_text.info(f"⏳ Minute {int(minutes_elapsed)+1}: Converting Masks to Geospatial Waypoints...")
                else:
                    status_text.info(f"⏳ Minute {int(minutes_elapsed)+1}: Finalizing Variable Rate Application (VRA) Map...")

                time.sleep(2) # Update interval

    else:
        st.error("Filename format incorrect. Use DJI_$$$$.JPG")
else:
    st.info("Upload a drone image to begin the 10-minute presentation pipeline.")