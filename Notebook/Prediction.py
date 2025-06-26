import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Embedding, LSTM, add
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.efficientnet import EfficientNetB0, preprocess_input

# ---------- Paths ----------
# WORKING_DIR = '/content/drive/MyDrive/archive2/working'
# IMAGES_DIR = '/content/drive/MyDrive/archive2/Images'
'.\.\Notebook\outputs\final_model.keras'
MODEL_WEIGHTS = r'.\.\Notebook\outputs\final_model.keras'  # path to weights (HDF5 or .keras)
TOKENIZER_PATH = r'C:\Users\shiha\OneDrive\Desktop\Logi_Prompt_Project\archive\Notebook\outputs\tokenizer.pkl'
max_length =37
# ----------------------------

# ---------- Load tokenizer and max_length ----------
with open(TOKENIZER_PATH, 'rb') as f:
    tokenizer = pickle.load(f)

# with open(MAXLEN_PATH, 'rb') as f:
#     max_length = pickle.load(f)

vocab_size = len(tokenizer.word_index) + 1

# ---------- Rebuild Captioning Model (same as training) ----------
def build_model(vocab_size, max_length):
    # Image input
    inputs1 = Input(shape=(1280,), name='image')
    fe1 = Dropout(0.5)(inputs1)
    fe2 = Dense(256, activation='relu')(fe1)

    # Text input
    inputs2 = Input(shape=(max_length,), name='text')
    se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
    se2 = Dropout(0.5)(se1)
    se3 = LSTM(256, use_cudnn=False)(se2)  # Must match training config

    # Merge and output
    decoder1 = add([fe2, se3])
    decoder2 = Dense(256, activation='relu')(decoder1)
    outputs = Dense(vocab_size, activation='softmax')(decoder2)

    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    return model

model = build_model(vocab_size, max_length)
model.load_weights(MODEL_WEIGHTS)

# ---------- Load feature extractor ----------
def load_feature_extractor():
    base_model = EfficientNetB0()
    model = Model(inputs=base_model.inputs, outputs=base_model.layers[-2].output)
    return model

feature_extractor = load_feature_extractor()

# ---------- Feature Extraction ----------
def extract_image_features(image_path, extractor):
    image = load_img(image_path, target_size=(224, 224))
    image = img_to_array(image)
    image = image.reshape((1, 224, 224, 3))
    image = preprocess_input(image)
    features = extractor.predict(image, verbose=0)
    return features

# ---------- Caption Prediction ----------
def convert_to_word(index, tokenizer):
    for word, idx in tokenizer.word_index.items():
        if idx == index:
            return word
    return None

def predict_caption(model, image, tokenizer, max_length):
    in_text = 'startseq'
    for _ in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        y_pred = model.predict([image, sequence], verbose=0)
        y_pred = np.argmax(y_pred)
        word = convert_to_word(y_pred, tokenizer)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'endseq':
            break
    return in_text

# ---------- Final Generation Function ----------
def generate_caption_for_image(image_path):
    features = extract_image_features(image_path, feature_extractor)
    caption = predict_caption(model, features, tokenizer, max_length)
    caption = caption.replace("startseq", "").replace("endseq", "").strip()

    image = Image.open(image_path)
    plt.imshow(image)
    plt.axis('off')
    plt.title(caption)
    plt.show()

# ---------- Run Example ----------
if __name__ == '__main__':
    test_image = os.path.join(r'C:\Users\shiha\OneDrive\Desktop\Logi_Prompt_Project\archive\Images\19212715_20476497a3.jpg')  # Replace as needed
    generate_caption_for_image(test_image)
# import importlib

# packages = {
#     'tensorflow': 'tensorflow',
#     'keras': 'keras',
#     'numpy': 'numpy',
#     'Pillow': 'PIL',
#     'matplotlib': 'matplotlib',
#     'nltk': 'nltk',
#     'sklearn': 'sklearn',  # Correct import name
#     'django': 'django',
# }

# for display_name, import_name in packages.items():
#     try:
#         lib = importlib.import_module(import_name)
#         print(f"{display_name}=={lib.__version__}")
#     except ImportError:
#         print(f"{display_name} not installed.")
#     except AttributeError:
#         print(f"{display_name} is installed but doesn't expose __version__.")
