# 🖼️ Image Caption Generator (CNN-LSTM)

## 📚 Project Overview

This project implements an image captioning model using a **Convolutional Neural Network (CNN)** and **Long Short-Term Memory (LSTM)** network.

- **EfficientNetB0** extracts visual features from images.
- **LSTM** generates textual captions based on the image features.
- Trained on the **Flickr8k dataset** — 8,091 images, each with 5 human-written captions.
- Includes a **Jupyter Notebook** for training/evaluation and a **Django web application** for deployment.

---

## ✨ Key Features

- ✅ Uses **EfficientNetB0** for efficient image feature extraction.
- ✅ **LSTM-based decoder** for caption generation.
- ✅ Supports training, inference, and deployment.
- ✅ **Django web app** with web UI + REST API for real-time captioning.

---

## ⚙️ Installation & Setup

### 📦 Prerequisites

- Python **3.10+**
- `pip` (Python package manager)
- `git`
- Access to the **Flickr8k dataset**

---

### 🔧 Setup Instructions

#### 1. Clone the Repository

```bash
git clone https://github.com/Shb987/Caption_Generator.git
cd Caption_Generator
2. Create Virtual Environment & Install Dependencies
bash
Copy
Edit
python -m venv venv
# Activate environment
source venv/bin/activate          # On macOS/Linux
venv\Scripts\activate             # On Windows

# Install packages
pip install -r requirements.txt
3. Download the Flickr8k Dataset
Download from the official Flickr8k dataset link.

Or use the Google Drive link provided (update instructions if needed).

Update the BASE_DIR variable in the notebook to point to your dataset location.

4. Prepare the Environment
bash
Copy
Edit
mkdir outputs
Ensure a GPU is available (optional) for faster training.

Run Django migrations:

bash
Copy
Edit
python manage.py migrate
🚀 Running the Server
1. Train the Model
Open the Jupyter Notebook:

bash
Copy
Edit
jupyter notebook notebooks/image-caption-generator-using-cnn-lstm.ipynb
Run all cells to:

Preprocess data

Extract image features

Train the model

Save weights to outputs/model.h5

⚠️ Precomputed files will be saved:

outputs/features.pkl

outputs/tokenizer.pkl

2. Start the Django Server
Ensure the following files are in the outputs/ directory:

model.h5

features.pkl

tokenizer.pkl

Run the server:

bash
Copy
Edit
python manage.py runserver
Visit: http://localhost:8000

3. Access the Interface
🌐 Web Interface: Upload an image to generate captions.

🧠 API Endpoint: POST /api/predict/
Example usage:

bash
Copy
Edit
curl -X POST -F 'image=@your_image.jpg' http://localhost:8000/api/predict/
📌 Notes
📁 Flickr8k dataset is not included in this repo.

💻 Use a GPU for faster training and inference.

🛠️ Django server is intended for development.
For production, use a WSGI server like Gunicorn or deploy on Heroku, AWS, etc.

Ensure outputs/ contains the necessary files before starting the server.

📄 License
This project is licensed under the MIT License.
See the LICENSE file for details.

🤝 Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

🙌 Acknowledgments
Thanks to the creators of the Flickr8k dataset and contributors to TensorFlow/Keras.

yaml
Copy
Edit

---

### ✅ Tip for Use:

- Save this as `README.md` in the root of your repo.
- If your GitHub repo has screenshots or a demo, consider adding:

```markdown
![Demo Screenshot](screenshots/demo.png)
