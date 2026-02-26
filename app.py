import streamlit as st
import time
import os
import re
import json
import folium
from streamlit_folium import st_folium
from folium.plugins import MarkerCluster
from PIL import Image

# --- CONFIGURATION ---
OUTPUT_IMAGE_DIR = "Output_Images"
VRA_OUTPUT_DIR = "VRA_Output"
SPRAYPOINTS_DIR = "Spraypoints"

st.set_page_config(page_title="Agricultural Weed Detection System", layout="wide")

st.title("🌱 Precision Agriculture: Weed Detection & VRA Mapping")
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

        if st.button("Start Processing Pipeline"):
            # --- MOCK PROCESSING TIMER (6 MINUTES) ---
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            total_seconds = 6 * 60 
            start_time = time.time()
            
            stages = [
                "Loading YOLOv8 + ResNet-50 Backbone...",
                "Running Graph Attention Network (GAT) for Occlusion Reasoning...",
                "Generating Precision Weed Masks...",
                "Mapping Spray Waypoints to GPS Coordinates...",
                "Finalizing Spatial GeoJSON Output..."
            ]

            while True:
                elapsed = time.time() - start_time
                percent = min(int((elapsed / total_seconds) * 100), 100)
                progress_bar.progress(percent)
                
                stage_idx = min(percent // 20, len(stages) - 1)
                status_text.info(f"**Current Task:** {stages[stage_idx]} ({percent}%)")
                
                if elapsed >= total_seconds:
                    break
                time.sleep(1) 
            
            status_text.success("✅ Analysis Complete!")
            st.balloons()

            # --- OUTPUT SEARCH & DISPLAY ---
            st.markdown("---")
            st.header("Analysis Results")
            
            col1, col2 = st.columns(2)
            pred_path = os.path.join(OUTPUT_IMAGE_DIR, f"DJI_{unique_id}_pred.png")
            vra_path = os.path.join(VRA_OUTPUT_DIR, f"graph_{unique_id}.png")
            geojson_path = os.path.join(SPRAYPOINTS_DIR, f"3_weed_map_{unique_id}.geojson")

            with col1:
                st.subheader("Detection Mask")
                if os.path.exists(pred_path):
                    st.image(pred_path, caption="Occlusion-Aware Prediction")
                else:
                    st.error("Prediction image not found.")

            with col2:
                st.subheader("VRA Graph Analysis")
                if os.path.exists(vra_path):
                    st.image(vra_path, caption="Graph Attention Network Output")
                else:
                    st.error("VRA Graph not found.")

            # --- GEOJSON MAP VISUALIZATION ---
            st.markdown("---")
            st.subheader("Interactive Field Map")
            
            if os.path.exists(geojson_path):
                with open(geojson_path, "r") as f:
                    geo_data = json.load(f)

                # Get the first coordinate to center the map
                try:
                    # Assuming GeoJSON contains Points
                    first_coords = geo_data['features'][0]['geometry']['coordinates']
                    # Folium uses [Lat, Lon], GeoJSON usually [Lon, Lat]
                    center_lat, center_lon = first_coords[1], first_coords[0]
                    
                    m = folium.Map(location=[center_lat, center_lon], zoom_start=18, tiles="CartoDB positron")
                    
                    # Add GeoJSON layer - handles many points efficiently
                    folium.GeoJson(
                        geo_data,
                        name="Weed Points",
                        tooltip=folium.GeoJsonTooltip(fields=[], aliases=[], localize=True),
                        marker=folium.CircleMarker(radius=3, color="red", fill=True, fill_color="red")
                    ).add_to(m)

                    st_folium(m, width=1200, height=500)
                    
                    st.download_button(
                        label="Download Full GeoJSON",
                        data=json.dumps(geo_data),
                        file_name=f"3_weed_map_{unique_id}.geojson",
                        mime="application/json"
                    )
                except Exception as e:
                    st.warning("Could not render map. Check GeoJSON coordinate format.")
            else:
                st.error("GeoJSON file not found.")

    else:
        st.error("Filename format incorrect. Use DJI_$$$$.JPG")
else:
    st.info("Upload a drone image to begin the 6-minute processing simulation.")