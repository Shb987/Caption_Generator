from django.shortcuts import render


def home(request):
    return render(request, 'home.html')
import os
import pickle
import numpy as np
from PIL import Image
from django.conf import settings

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import default_storage

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Embedding, LSTM, add
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.efficientnet import EfficientNetB0, preprocess_input

# ---------- Load Tokenizer ----------
token_path = os.path.join(settings.BASE_DIR, '..', 'Notebook', 'outputs', 'tokenizer.pkl')

TOKENIZER_PATH = token_path
print(TOKENIZER_PATH)
print("BASE_DIR:", settings.BASE_DIR)

with open(TOKENIZER_PATH, 'rb') as f:
    tokenizer = pickle.load(f)

max_length = 37
vocab_size = len(tokenizer.word_index) + 1

# ---------- Build Model ----------
def build_model(vocab_size, max_length):
    inputs1 = Input(shape=(1280,))
    fe1 = Dropout(0.5)(inputs1)
    fe2 = Dense(256, activation='relu')(fe1)

    inputs2 = Input(shape=(max_length,))
    se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
    se2 = Dropout(0.5)(se1)
    se3 = LSTM(256)(se2)

    decoder1 = add([fe2, se3])
    decoder2 = Dense(256, activation='relu')(decoder1)
    outputs = Dense(vocab_size, activation='softmax')(decoder2)

    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    return model

model = build_model(vocab_size, max_length)
model_path = os.path.join(settings.BASE_DIR, '..', 'Notebook', 'outputs', 'best_model.keras')
print("Resolved model path:", os.path.abspath(model_path))

model.load_weights(os.path.abspath(model_path))


# ---------- Feature Extractor ----------
def load_feature_extractor():
    base_model = EfficientNetB0()
    model = Model(inputs=base_model.inputs, outputs=base_model.layers[-2].output)
    return model

feature_extractor = load_feature_extractor()

# ---------- Helper Functions ----------
def extract_image_features(image_path, extractor):
    image = load_img(image_path, target_size=(224, 224))
    image = img_to_array(image)
    image = image.reshape((1, 224, 224, 3))
    image = preprocess_input(image)
    features = extractor.predict(image, verbose=0)
    return features

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

# ---------- Django View ----------
@csrf_exempt
def predict_caption_api(request):
    if request.method == 'POST' and request.FILES.get('image'):
        image_file = request.FILES['image']
        image_path = default_storage.save(f"temp/{image_file.name}", image_file)

        full_path = os.path.join(default_storage.location, image_path)
        features = extract_image_features(full_path, feature_extractor)
        raw_caption = predict_caption(model, features, tokenizer, max_length)
        caption = raw_caption.replace("startseq", "").replace("endseq", "").strip()

        return JsonResponse({'caption': caption})

    return JsonResponse({'error': 'Please POST an image file'}, status=400)

