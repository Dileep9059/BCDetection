import os
import gdown
from flask import Flask, render_template, request
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tf_keras.models import load_model
from tf_keras.preprocessing.image import img_to_array, load_img

from PIL import Image, ImageEnhance
import numpy as np
import random



app = Flask(__name__)

# --- Cloud Model Download Logic ---
MODEL_PATH = "ensemble_model.h5"
# REPLACE the string below with your Google Drive File ID
GDRIVE_FILE_ID = "YOUR_GOOGLE_DRIVE_FILE_ID_HERE"

if not os.path.exists(MODEL_PATH):
    print(f"Model not found locally. Downloading from Google Drive...")
    gdown.download(id=GDRIVE_FILE_ID, output=MODEL_PATH, quiet=False)

model = load_model(MODEL_PATH, compile=False)

def predict_label(img_path):
    img = load_img(img_path, target_size=(128, 128))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)

    # Image enhancement
    image = Image.fromarray(np.uint8(img[0]))
    image = ImageEnhance.Brightness(image).enhance(random.uniform(0.8, 1.2))
    image = ImageEnhance.Contrast(image).enhance(random.uniform(0.8, 1.2))
    image = ImageEnhance.Sharpness(image).enhance(random.uniform(0.8, 1.2))
    img = np.array(image) / 255.0

    
    predictions = np.argmax(model.predict(np.expand_dims(img, axis=0), verbose=0))

    return predictions




@app.route('/train_reslt', methods=['GET', 'POST'])
def train_reslt():
	return render_template('results.html')

# routes
@app.route("/", methods=['GET', 'POST'])
def main():
	return render_template("index.html")

def is_valid_ultrasound(img_path):
    img = cv2.imread(img_path)
    if img is None: return False
    
    b, g, r = cv2.split(img)
    rg_diff = np.mean(np.abs(r.astype(int) - g.astype(int)))
    gb_diff = np.mean(np.abs(g.astype(int) - b.astype(int)))
    
    if rg_diff > 15 or gb_diff > 15:
        return False
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if np.sum(gray < 25) / gray.size < 0.05:
        return False
        
    return True

@app.route("/submit", methods = ['GET', 'POST'])
def get_output():
	if request.method == 'POST':
		img = request.files['my_image']

		img_path = "static/" + img.filename	
		img.save(img_path)

		dataset_labels = {'Benign':0, 'Malignant':1, 'Normal':2}

		def getlabel(n):
			for x, y in dataset_labels.items(): 
				if n==y: return x

		if not is_valid_ultrasound(img_path):
			cls_name = "Invalid: Not an Ultrasound"
		else:
			class_ = predict_label(img_path)
			cls_name = getlabel(class_)

	return render_template("index.html", prediction = cls_name, img_path = img_path)


if __name__ =='__main__':
	#app.debug = True
	app.run(debug = True)
