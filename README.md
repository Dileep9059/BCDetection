# Breast Cancer Detection AI (BCDetection)

A deep learning medical imaging application built with Flask and TensorFlow (Keras) that classifies breast ultrasound scans into three categories: **Benign**, **Malignant**, and **Normal**. 

It features an ensemble Convolutional Neural Network (CNN) model and a built-in OpenCV anomaly detection pipeline to automatically reject non-ultrasound images (e.g. selfies or random colorful photos).

## Features
- **AI Classification**: Fast prediction of breast ultrasound scans.
- **Anomaly Detection**: OpenCV-based filtering rejects non-grayscale and non-ultrasound formatted images before they hit the ML model.
- **Modern UI**: Dark-mode, responsive web interface with drag-and-drop functionality.

---

## How to Run Locally

If you have just cloned this repository from GitHub, follow these steps to run it on your own machine.

### 1. Prerequisites
- **Python 3.9 - 3.11** installed.
- Because the `ensemble_model.h5` file (the trained AI brain) is very large (~111MB), it is **not** included in this repository. You must obtain the `ensemble_model.h5` file separately and place it in the root folder of this project before running.

### 2. Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Dileep9059/BCDetection.git
   cd BCDetection
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv env
   ```

3. **Activate the virtual environment:**
   - **Windows:** `.\env\Scripts\activate`
   - **Mac/Linux:** `source env/bin/activate`

4. **Install the dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

5. **Set Legacy Keras Flag (Important for TensorFlow 2.16+):**
   - **Windows:** `$env:TF_USE_LEGACY_KERAS="1"`
   - **Mac/Linux:** `export TF_USE_LEGACY_KERAS="1"`

6. **Run the Flask application:**
   ```bash
   python app.py
   ```

7. **Open your browser:**
   Navigate to `http://127.0.0.1:5000` to access the application!