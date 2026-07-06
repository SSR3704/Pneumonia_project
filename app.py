
import os
import gdown
import numpy as np
from PIL import Image
import cv2
import tensorflow as tf
from datetime import datetime

from flask import Flask, request, render_template, send_file
from werkzeug.utils import secure_filename

from tensorflow.keras.models import load_model

# PDF
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas as pdf_canvas
from reportlab.lib import colors
from reportlab.lib.units import inch

import matplotlib
matplotlib.use('Agg')

# Download model if not present
MODEL_PATH = "model_weights/vgg19_model_01.h5"
if not os.path.exists(MODEL_PATH):
    os.makedirs('model_weights', exist_ok=True)
    gdown.download(
        'https://drive.google.com/uc?id=1m7wdnbRmtVoC9VPbM-55cbqGzDTBUqaq',
        MODEL_PATH,
        quiet=False
    )

# Load model
model = load_model(MODEL_PATH)

# Load model
model = load_model("model_weights/vgg19_model_01.h5")

# Get last conv layer
last_conv_layer = None
for layer in reversed(model.layers):
    if isinstance(layer, tf.keras.layers.Conv2D):
        last_conv_layer = layer.name
        break

print(f"Using layer: {last_conv_layer}")

app = Flask(__name__)
print("Model loaded. Check localhost")


# ============== HELPER FUNCTIONS ==================

def get_className(classNo):
    if classNo == 0:
        return "Normal"
    return "Pneumonia"


def generate_gradcam(img_path):
    # Read and preprocess
    image = cv2.imread(img_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (128, 128))
    image_array = np.array(image_resized) / 255.0
    input_img = np.expand_dims(image_array, axis=0)

    # Grad model
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer).output, model.output]
    )

    # Compute gradients
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(input_img)
        pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    heatmap = heatmap.numpy()

    # Overlay
    h, w = image_rgb.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    superimposed = cv2.addWeighted(image_rgb, 0.6, heatmap_colored, 0.4, 0)

    # Save gradcam
    gradcam_path = os.path.join(os.path.dirname(img_path), 'gradcam_result.jpg')
    Image.fromarray(superimposed).save(gradcam_path)

    # Confidence score
    confidence = float(predictions[0][int(pred_index)]) * 100

    return gradcam_path, int(pred_index), round(confidence, 2)


def generate_pdf(patient_name, result, confidence, img_path, gradcam_path):
    pdf_path = os.path.join(os.path.dirname(img_path), 'report.pdf')
    c = pdf_canvas.Canvas(pdf_path, pagesize=A4)
    width, height = A4

    # Header background
    c.setFillColor(colors.HexColor('#0d3d6e'))
    c.rect(0, height - 100, width, 100, fill=True, stroke=False)

    # Title
    c.setFillColor(colors.white)
    c.setFont("Helvetica-Bold", 22)
    c.drawCentredString(width / 2, height - 45, "PNEUMONIA DETECTION REPORT")
    c.setFont("Helvetica", 11)
    c.drawCentredString(width / 2, height - 70, "Powered by VGG19 Deep Learning Model")

    # Date
    c.setFillColor(colors.HexColor('#0d3d6e'))
    c.setFont("Helvetica", 11)
    c.drawString(50, height - 130, f"Date: {datetime.now().strftime('%d %B %Y  |  %I:%M %p')}")
    c.drawString(50, height - 150, f"Patient Name: {patient_name}")

    # Divider
    c.setStrokeColor(colors.HexColor('#1a5fa8'))
    c.setLineWidth(1.5)
    c.line(50, height - 165, width - 50, height - 165)

    # Result box
    if result == "Pneumonia":
        box_color = colors.HexColor('#fdecea')
        text_color = colors.HexColor('#c0392b')
        result_text = "⚠ PNEUMONIA DETECTED"
    else:
        box_color = colors.HexColor('#e8f8f0')
        text_color = colors.HexColor('#1a7a45')
        result_text = "✓ NORMAL - NO PNEUMONIA"

    c.setFillColor(box_color)
    c.roundRect(50, height - 230, width - 100, 50, 8, fill=True, stroke=False)
    c.setFillColor(text_color)
    c.setFont("Helvetica-Bold", 16)
    c.drawCentredString(width / 2, height - 200, result_text)

    # Confidence
    c.setFillColor(colors.HexColor('#1a5fa8'))
    c.setFont("Helvetica-Bold", 13)
    c.drawCentredString(width / 2, height - 255, f"Confidence Score: {confidence}%")

    # Images
    c.setFont("Helvetica-Bold", 12)
    c.setFillColor(colors.HexColor('#0d3d6e'))
    c.drawString(50, height - 290, "Original X-Ray:")
    c.drawString(width / 2 + 10, height - 290, "Grad-CAM Analysis:")

    img_display_width = (width - 120) / 2

    try:
        c.drawImage(img_path, 50, height - 530, 
                   width=img_display_width, height=220, preserveAspectRatio=True)
    except:
        pass

    try:
        c.drawImage(gradcam_path, width / 2 + 10, height - 530,
                   width=img_display_width, height=220, preserveAspectRatio=True)
    except:
        pass

    # Grad-CAM legend
    c.setFont("Helvetica", 10)
    c.setFillColor(colors.HexColor('#888888'))
    c.drawCentredString(width / 2, height - 550,
                       "Red/Yellow areas = Region where model focused to make prediction")

    # Divider
    c.setStrokeColor(colors.HexColor('#1a5fa8'))
    c.line(50, height - 570, width - 50, height - 570)

    # Disclaimer
    c.setFont("Helvetica-Oblique", 9)
    c.setFillColor(colors.HexColor('#aaaaaa'))
    c.drawCentredString(width / 2, height - 590,
                       "This report is generated by an AI model and is for educational purposes only.")
    c.drawCentredString(width / 2, height - 605,
                       "Please consult a certified medical professional for diagnosis.")

    c.save()
    return pdf_path


# ============== FLASK ROUTES ==================

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def upload():
    if request.method == 'POST':
        try:
            f = request.files['file']
            patient_name = request.form.get('patient_name', 'Unknown')

            base_path = os.path.dirname(__file__)
            uploads_dir = os.path.join(base_path, 'uploads')
            os.makedirs(uploads_dir, exist_ok=True)

            file_path = os.path.join(uploads_dir, secure_filename(f.filename))
            f.save(file_path)

            gradcam_path, pred_index, confidence = generate_gradcam(file_path)
            result = get_className(pred_index)

            # Generate PDF
            generate_pdf(patient_name, result, confidence, file_path, gradcam_path)

            return render_template(
                'index.html',
                prediction=result,
                confidence=confidence,
                gradcam_image='/gradcam',
                patient_name=patient_name,
                show_pdf=True
            )

        except Exception as e:
            print(f"ERROR: {e}")
            return render_template('index.html', prediction=f"Error: {str(e)}")

    return render_template('index.html', prediction=None)


@app.route('/gradcam')
def serve_gradcam():
    base_path = os.path.dirname(__file__)
    gradcam_path = os.path.join(base_path, 'uploads', 'gradcam_result.jpg')
    return send_file(gradcam_path, mimetype='image/jpeg')


@app.route('/download_report')
def download_report():
    base_path = os.path.dirname(__file__)
    pdf_path = os.path.join(base_path, 'uploads', 'report.pdf')
    return send_file(pdf_path, as_attachment=True, download_name='Pneumonia_Report.pdf')


# ================================

if __name__ == '__main__':
    app.run(debug=True)
