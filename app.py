import os
import json
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
# image_processor 임포트 (수정된 함수 사용)
from utils.image_processor import preprocess_image

# PyTorch 임포트
import torch
import torch.nn as nn # 모델 클래스 정의에 필요할 수 있음
import numpy as np # 결과 처리 등에 여전히 사용될 수 있음

app = Flask(__name__)

# --- 설정 ---
MODEL_DIR = 'model'
MODEL_FILENAME = 'goldendoodle.pth' # <<< 모델 파일 이름 변경됨
LABELS_FILENAME = 'labels.json'

MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)
LABELS_PATH = os.path.join(MODEL_DIR, LABELS_FILENAME)

# --- 모델 및 레이블 로드 ---
model = None
labels = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # GPU 사용 가능하면 사용
print(f"Using device: {device}")

# ==============================================================================
# !! 중요 !! 실제 사용하는 PyTorch 모델의 클래스 정의가 필요합니다.
# 아래는 예시이며, 실제 모델 구조에 맞게 클래스를 정의하거나 import 해야 합니다.
# ==============================================================================
class YourDogBreedModel(nn.Module): # 예시 클래스 이름
    def __init__(self, num_classes):
        super(YourDogBreedModel, self).__init__()
        # --- 실제 모델 아키텍처 정의 ---
        # 예시: 사전 학습된 모델 사용 (torchvision 활용)
        # import torchvision.models as models
        # self.base_model = models.resnet50(pretrained=True) # 또는 weights=ResNet50_Weights.IMAGENET1K_V1
        # num_ftrs = self.base_model.fc.in_features
        # self.base_model.fc = nn.Linear(num_ftrs, num_classes) # 마지막 레이어 변경

        # 예시: 간단한 CNN 모델
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(16 * 112 * 112, num_classes) # 입력 크기에 따라 변경 필요 (224x224 기준)

    def forward(self, x):
        # --- 실제 모델의 forward pass 정의 ---
        # 예시: 사전 학습된 모델 사용
        # return self.base_model(x)

        # 예시: 간단한 CNN 모델
        x = self.pool(self.relu(self.conv1(x)))
        x = x.view(x.size(0), -1) # Flatten
        x = self.fc1(x)
        return x

# ==============================================================================

def load_pytorch_model_and_labels():
    """PyTorch 모델과 레이블을 로드하는 함수"""
    global model, labels
    try:
        # 레이블 파일 로드
        if os.path.exists(LABELS_PATH):
            with open(LABELS_PATH, 'r', encoding='utf-8') as f:
                labels = json.load(f)
            print(f"Labels loaded successfully from {LABELS_PATH}")
            if not isinstance(labels, list):
                print("Warning: Labels file does not contain a list.")
                labels = None
        else:
            print(f"Error: Labels file not found at {LABELS_PATH}")
            return # 레이블 없으면 모델 로드 의미 없음

        # 모델 로드
        if labels and os.path.exists(MODEL_PATH):
            num_classes = len(labels)

            # !! 중요 !! 모델 클래스 인스턴스 생성
            # YourDogBreedModel을 실제 모델 클래스 이름으로 바꾸세요.
            loaded_model_structure = YourDogBreedModel(num_classes=num_classes)

            # 저장된 state_dict 로드 (CPU 또는 GPU 맵핑)
            state_dict = torch.load(MODEL_PATH, map_location=device)

            # state_dict 를 모델 구조에 로드
            # DataParallel 등으로 저장된 경우 키 이름 조정 필요할 수 있음 (예: 'module.' 접두사 제거)
            if isinstance(state_dict, dict) and any(key.startswith('module.') for key in state_dict.keys()):
                 state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

            loaded_model_structure.load_state_dict(state_dict)

            # 모델을 device(CPU/GPU)로 이동 및 평가 모드로 설정
            model = loaded_model_structure.to(device)
            model.eval() # <<< 평가 모드 설정 (매우 중요!)
            print(f"PyTorch Model loaded successfully from {MODEL_PATH} to {device}")
        else:
            if not labels:
                 print("Model loading skipped: Labels not loaded.")
            if not os.path.exists(MODEL_PATH):
                 print(f"Error: Model file not found at {MODEL_PATH}")

    except Exception as e:
        print(f"Error loading PyTorch model or labels: {e}")
        model = None
        labels = None

# 앱 시작 시 모델 로드 시도
load_pytorch_model_and_labels()

@app.route('/predict', methods=['POST'])
def predict():
    """이미지를 받아 품종 예측 결과를 반환하는 API 엔드포인트 (PyTorch)"""
    if model is None or labels is None:
        print("Error: Model or labels not loaded properly.")
        return jsonify({"error": "Model or labels not available"}), 500

    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    try:
        img_bytes = file.read()
        # PyTorch용 전처리 함수 사용
        input_tensor = preprocess_image(img_bytes)

        if input_tensor is None:
            print("Error during image preprocessing step.")
            return jsonify({"error": "Image preprocessing failed"}), 400

        # --- PyTorch 모델 예측 수행 ---
        input_tensor = input_tensor.to(device) # 텐서를 모델과 같은 device로 이동

        with torch.no_grad(): # <<< 그래디언트 계산 비활성화 (추론 시 필수)
            outputs = model(input_tensor) # 모델 포워드 패스

            # 모델 출력(logits)을 확률로 변환 (예: Softmax)
            # 모델 마지막 레이어에 Softmax가 이미 포함되어 있다면 이 단계 생략 가능
            probabilities = torch.softmax(outputs, dim=1)[0] # 배치 차원 제거

            # 확률값을 CPU로 이동하고 NumPy 배열로 변환 (결과 처리 용이)
            probabilities = probabilities.cpu().numpy()

        # --- 결과 처리 ---
        top_n = 3
        top_indices = probabilities.argsort()[-top_n:][::-1]

        results = []
        for i in top_indices:
            if i < len(labels):
                breed_name = labels[i]
                confidence = float(probabilities[i]) * 100
                results.append({
                    "breedName": breed_name,
                    "confidence": round(confidence, 2)
                })
            else:
                print(f"Warning: Predicted index {i} is out of bounds.")

        print(f"Prediction results for '{file.filename}': {results}")
        return jsonify(results)

    except Exception as e:
        print(f"Error during prediction process: {e}")
        return jsonify({"error": "Prediction failed due to an internal error"}), 500

# 서버 상태 확인용 엔드포인트 (선택적)
@app.route('/health', methods=['GET'])
def health_check():
    status = {
        "status": "OK",
        "model_loaded": model is not None,
        "labels_loaded": labels is not None,
        "device": str(device)
    }
    return jsonify(status)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True) # 디버그 모드는 개발 시에만 사용