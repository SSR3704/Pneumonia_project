\# 🫁 Pneumonia Detection Web App



An AI-powered web application that detects pneumonia from chest X-ray images using a \*\*VGG19 deep learning model\*\*. It also generates \*\*Grad-CAM heatmaps\*\* to visualize which regions of the X-ray influenced the model's prediction, and produces a downloadable \*\*PDF report\*\*.



\---



\## 🚀 Features



\- 📤 Upload chest X-ray images directly from the browser

\- 🤖 Pneumonia detection using a fine-tuned \*\*VGG19\*\* model

\- 🔥 \*\*Grad-CAM visualization\*\* — highlights the areas the model focused on

\- 📄 Auto-generated \*\*PDF report\*\* with patient name, result, confidence score, and heatmap

\- ⚡ Built with \*\*Flask\*\* for a lightweight and fast web interface



\---



\## 🖥️ Tech Stack



| Layer | Technology |

|-------|------------|

| Backend | Python, Flask |

| Deep Learning | TensorFlow, Keras (VGG19) |

| Image Processing | OpenCV, Pillow |

| PDF Generation | ReportLab |

| Frontend | HTML, CSS (Jinja2 templates) |



\---



\## 📁 Project Structure



```

pneumonia\_project/

│

├── app.py                  # Main Flask application

├── requirements.txt        # Python dependencies

├── .gitignore

│

├── templates/

│   └── index.html          # Frontend UI

│

├── uploads/                # Temporary storage for uploaded X-rays

│

└── model\_weights/          # (Not included - download separately)

&#x20;   └── vgg19\_model\_01.h5

```



\---



\## ⬇️ Setup \& Installation



\### 1. Clone the repository

```bash

git clone https://github.com/soumyojit868/Pneumonia\_project.git

cd Pneumonia\_project

```



\### 2. Create a virtual environment

```bash

python -m venv venv



\# Activate it:

\# Windows:

venv\\Scripts\\activate

\# Mac/Linux:

source venv/bin/activate

```



\### 3. Install dependencies

```bash

pip install -r requirements.txt

```



\### 4. Download the Model Weights

The model file is too large for GitHub. Download it from the link below and place it inside the `model\_weights/` folder:



📥 \*\*\[Download vgg19\_model\_01.h5 from Google Drive](#)\*\* ← \*(Replace with your actual link)\*



```

model\_weights/

└── vgg19\_model\_01.h5

```



\### 5. Run the app

```bash

python app.py

```



Then open your browser and go to: \*\*http://localhost:5000\*\*



\---



\## 🧪 How to Use



1\. Open the web app in your browser

2\. Enter the \*\*patient name\*\*

3\. Upload a \*\*chest X-ray image\*\*

4\. Click \*\*Predict\*\*

5\. View the result, confidence score, and Grad-CAM heatmap

6\. Download the \*\*PDF report\*\*



\---



\## 📊 Model Info



\- \*\*Architecture:\*\* VGG19 (Transfer Learning)

\- \*\*Input Size:\*\* 128 × 128 pixels

\- \*\*Classes:\*\* Normal | Pneumonia

\- \*\*Explainability:\*\* Grad-CAM on last Conv2D layer



\---



\## ⚠️ Disclaimer



> This application is built for \*\*educational purposes only\*\*.

> It is \*\*not a substitute\*\* for professional medical diagnosis.

> Always consult a certified medical professional.



\---



\## 👨‍💻 Author



\*\*Soumyojit\*\* — \[GitHub](https://github.com/soumyojit868)

