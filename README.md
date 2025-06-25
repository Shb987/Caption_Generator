Image Caption Generator (CNN-LSTM)
Project Overview
This project implements an image captioning model using a Convolutional Neural Network (CNN) and Long Short-Term Memory (LSTM) network. The CNN (EfficientNetB0) extracts features from images, and the LSTM generates textual captions. The model is trained on the Flickr8k dataset, which contains 8,091 images, each with five captions. The project includes a Jupyter Notebook for training and evaluation, and a Django web application for deploying the model to generate captions via a web interface or API.
Key features:

Uses EfficientNetB0 for efficient image feature extraction.
LSTM-based decoder for sequence generation.
Supports training, inference, and deployment.
Deployed as a Django app with a web interface and API endpoint for caption generation.

Installation & Setup Instructions
Prerequisites

Python 3.10  or higher
pip (Python package manager)
Git
Access to the Flickr8k dataset (download instructions below)

Steps 

1. Clone the repository:
git clone https://github.com/Shb987/Caption_Generator.git
cd Caption_Generator


2. Install dependencies:Create a virtual environment (optional but recommended) and install required packages:
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt



3. Download the Flickr8k dataset:

Download the dataset from the drive.
Alternatively, update the BASE_DIR variable in the notebook to point to your dataset location.


4. Prepare the environment:

Ensure a GPU is available for faster training (optional; TensorFlow will use CPU otherwise).
Create an outputs/ folder for model weights and features:mkdir outputs


Apply Django migrations for the server:python manage.py migrate





How to Run the Server
The project includes a Django application (generatorapp) to serve the trained model for generating captions from uploaded images.

1. Train the model :

Open the Jupyter Notebook:jupyter notebook notebooks/image-caption-generator-using-cnn-lstm.ipynb


Run all cells to preprocess data, extract features, train the model, and save weights to outputs/model.h5.
Note: Precomputed features and tokenizer are saved as outputs/features.pkl and outputs/tokenizer.pkl.


2. Start the Django server:

Ensure the model weights, features, and tokenizer are in the outputs/ folder.
Run the Django development server:python manage.py runserver


The server will start on http://localhost:8000.


3. Access the server:

Open a web browser and navigate to http://localhost:8000 to access the upload interface.
Use the /api/predict/ endpoint for API-based caption generation (see Example Usage).







Notes

The Flickr8k dataset is not included in the repository due to its size. Follow the download instructions above.
Model training requires significant computational resources; use a GPU for faster results.
The Django server is for development. For production, use a WSGI server like Gunicorn and deploy on a cloud platform (e.g., Heroku, AWS).
Ensure the outputs/ folder contains the trained model and precomputed files before running the server.

License
This project is licensed under the MIT License. See the LICENSE file for details.
