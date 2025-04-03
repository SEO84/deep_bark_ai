from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import json
import os

app = Flask(__name__)

# 모델 및 레이블 로드
MODEL_PATH = os.path.join('model', 'dog_breed_model.h5')
LABELS_PATH = os.path.join('model', 'labels.json')

try:
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(LABELS_PATH, 'r', encoding='utf-8') as f:
        labels = json.load(f)
    print("Model and labels loaded successfully.")
except Exception as e:
    print(f"Error loading model or labels: {e}")
    model = None
    labels = None

def preprocess_image(image_bytes):
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        img = img.resize((224, 224))  # 모델 입력 크기에 맞게 조절
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        print(f"Error in preprocess_image: {e}")
        return None

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or labels is None:
        return jsonify({'error': 'Model or labels not loaded'}), 500

    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    file = request.files['image']
    image_bytes = file.read()
    processed_image = preprocess_image(image_bytes)
    if processed_image is None:
        return jsonify({'error': 'Image processing failed'}), 400

    try:
        predictions = model.predict(processed_image)[0]  # 배치 차원 제거
        top_n = 3
        top_indices = np.argsort(predictions)[-top_n:][::-1]
        results = []
        for i in top_indices:
            # labels가 dict인 경우 인덱스 문자열로 접근, 리스트면 정수 인덱스
            breed_name = labels[str(i)] if isinstance(labels, dict) else labels[i]
            confidence = float(predictions[i]) * 100
            results.append({
                'breed_name': breed_name,
                'confidence': round(confidence, 2)
            })
        return jsonify(results)
    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({'error': 'Prediction failed'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
