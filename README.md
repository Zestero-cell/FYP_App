# Variable-Rate-Chemical-Dispersal-Application: Occlusion-Aware Weed Detection & VRA System

This repository contains the application for a Final Year Project focused on an occlusion-aware weed detection system for banana plantations. Developed in collaboration with **ICAR-NRCB Trichy**, the system uses drone imagery to identify weeds and calculate precise spray points for Variable Rate Application (VRA).

## Features
- **Weed Detection:** Processes drone-captured imagery to identify and segment weeds within banana plantations.
- **Occlusion Awareness:** Implements specialized algorithms to handle overlapping leaves and environmental occlusions.
- **VRA Output Generation:** Automatically calculates Variable Rate Application (VRA) maps and coordinates.
- **Spraypoint Visualization:** Generates precise spray points saved in specialized formats for agricultural machinery or drones.
- **Intermediate Processing:** Accessible logs and images of the computer vision pipeline (segmentation, filtering, etc.).

## Repository Structure
```text
├── app.py                # Main application entry point (Streamlit/Flask-based)
├── requirements.txt      # Python dependencies
├── Intermediate_Images/  # Visualizes steps in the CV pipeline
├── Output_Images/        # Final detection and segmentation results
├── Spraypoints/          # Generated coordinate files for spraying
├── VRA_Output/           # Variable Rate Application data and maps
└── README.md
```

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Zestero-cell/Variable-Rate-Chemical-Dispersal-Application.git
   cd Variable-Rate-Chemical-Dispersal-Application
   ```

2. **Set up a virtual environment (Optional but recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the main application using:
```bash
python app.py
```
*(If the app uses Streamlit, use `streamlit run app.py` instead.)*

1. Upload drone imagery of the plantation.
2. The system will process the images through the occlusion-aware model.
3. View the detected weeds in `Output_Images/`.
4. Download the spray points and VRA maps from the respective folders.

## Technologies Used
- **Language:** Python
- **Computer Vision:** OpenCV, PyTorch/TensorFlow (for model inference)
- **Data Handling:** NumPy, Pandas
- **Application Framework:** (e.g., Streamlit/Flask)

## Acknowledgments
Special thanks to **ICAR-National Research Centre for Banana (NRCB), Trichy** for providing the datasets and domain expertise for this research project.

---
*Created as part of the Final Year Project on Agricultural Automation.*
