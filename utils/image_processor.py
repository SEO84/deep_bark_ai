from PIL import Image
import io
import torch # torch 임포트
import torchvision.transforms as transforms # torchvision 임포트

# PyTorch 모델 (예: ImageNet 사전 학습 모델)에서 흔히 사용하는 전처리
# !! 중요 !! 실제 사용하는 모델의 학습 시 적용된 전처리와 동일하게 맞춰야 합니다.
def preprocess_image_pytorch(image_bytes, target_size=(224, 224)):
    """
    이미지 바이트를 받아 PyTorch 모델 입력 형식에 맞게 전처리합니다.

    Args:
        image_bytes: 이미지 파일의 바이트 데이터
        target_size: 모델이 요구하는 이미지 크기 (튜플 형태, 예: (224, 224))

    Returns:
        전처리된 PyTorch 텐서 또는 None (오류 발생 시)
    """
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB') # RGB로 변환

        # 일반적인 PyTorch 이미지 변환 파이프라인 정의
        preprocess_transform = transforms.Compose([
            transforms.Resize(target_size), # 리사이즈
            transforms.ToTensor(),          # PIL Image -> PyTorch Tensor (0-1 값으로 자동 스케일링)
            # ImageNet 표준 정규화 (모델 학습 시 사용한 값으로 변경 필요)
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # 변환 적용
        tensor = preprocess_transform(img)

        # 배치 차원 추가 (모델이 배치 입력을 받는 경우)
        tensor = tensor.unsqueeze(0) # (C, H, W) -> (1, C, H, W)

        return tensor

    except Exception as e:
        print(f"Error during PyTorch image preprocessing: {e}")
        return None

# 이전 함수 이름과의 호환성을 위해 유지하거나 이름을 변경하여 사용
preprocess_image = preprocess_image_pytorch