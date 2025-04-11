## Overview
본 실험의 목적은 VGG 논문에서 제안된 모델 구조와 학습 기법을 CIFAR-10 데이터셋에 맞춰 재현하고 적용 가능성과 성능을 평가하는 것이다.
CIFAR-10은 ImageNet과 달리 이미지의 해상도가 낮고 클래스 수가 작은 소규모 데이터셋으로 원 논문과는 다른 특성을 가진다.
이에 따라 모델의 아키텍쳐와 학습 전략, 평가 방식을 조정했으며, 특히 입력 이미지의 해상도 차이를 반영해 Convolution 레이어의 채널 수를 축소했다.
실험 전반에 걸쳐 가능한 한 논문의 설정을 유지하되, CIFAR-10에 맞춘 실용적인 변형과 해석을 통해 성능 재현을 시도하였다.

## Model Architecture
Convolution 레이어의 채널 수는 입력 데이터로부터 유용한 feature를 추출하는데 핵심적인 모델의 특성이다.
ImageNet에서 CIFAR-10으로 데이터셋이 변경되며 입력 이미지의 해상도가 축소되었고 추출해야할 feature 양도 줄어들었을 것이다.
모델이 새로운 데이터셋을 잘 학습할 수 있도록 Convolution 레이어의 채널 수를 줄여 모델의 표현력을 적절히 축소 조정했다.
시행착오를 통해 논문에 제시된 VGG 구조에서 Convolution 레이어의 채널 수를 1/8로 축소한 모델이 CIFAR-10 데이터셋에서 좋은 성능을 보였고,
이는 CIFAR-10과 ImageNet 간의 이미지 해상도 비율이 1/7(=32/224)와 유사하다.
본 실험에서는 이러한 분석을 바탕으로 논문에 제시된 VGG 구조에서 Convolution 레이어의 채널 수를 1/8로 축소한 모델을 이용했다.

## Training
훈련 데이터셋은 논문에 제시된 방식에 최대한 맞춰 다음과 같은 전처리를 적용했다.

  1. 이미지의 짧은 변의 길이가 $S$가 되도록 Resize
  2. 모델 입력 크기인 $32\times32$로 Random Crop
  3. Colot Jitter 적용
  4. Normalize 적용

이때 $S$의 선택 방식에 따라 두 가지 변환으로 나누어 사용했다.
  - Fix-Sacle Transform: 고정된 S를 이용
  - Multi-Scale Transform: $[S_{min}, S_{max}]$에서 무작위로 $S$를 샘플링

학습에 이용된 하이퍼 파라미터와 설정은 다음과 같다.
- Batch Size: 32
- Optimizer: SGD with `learning rate = 0.01`, `momentum = 0.9`
- Scheduler: Validation Accuracy가 개선되지 않으면 `learning rate`을 1/10으로 축소

## Testing
반복적인 Convolution 연산을 거치며 출력 feature map의 해상도가 1x1로 축소되어 논문에서 이용된 Dense Evaluation은 적용할 수 없었다.
대신 논문에서 이용된 Test-time augmentation으로 Horizontal Flip을 적용했다.
원본과 반전된 이미지 각각에 대해 Softmax 출력을 계산한 뒤, 두 결과의 평균을 최종 예측 확률로 이용했다.

논문에서는 평가 지표로 Top-1, Top-5 Error를 사용했지만, 데이터셋이 CIFAR-10으로 변경됨에 따라 클래스의 수가 10개로 줄어 지표를 그대로 적용하기 어려웠다.
따라서 본 실험에서는 Accuracy만을 모델 성능 평가 지표로 이용했다.

## Results
<div align="center">
  <img src="https://github.com/user-attachments/assets/07631516-a397-43f2-8ff8-756a3d15a6c3" width="70%">
  <p>Fig.1 - TinyVGG A, Fix-Scale(s=36) Training Log</p>
</div>

<div align="center">
  <img src="https://github.com/user-attachments/assets/192c35d1-01f0-4f7f-bab6-c202b1355df6" width="70%">
  <p>Fig.2 - TinyVGG B, Fix-Scale(s=36) Training Log</p>
</div>

<div align="center">
  <img src="https://github.com/user-attachments/assets/587bd06b-5836-4dbe-8f65-6f633596b03c" width="70%">
  <p>Fig.3 - TinyVGG C, Fix-Scale(s=36) Training Log</p>
</div>

<div align="center">
  <img src="https://github.com/user-attachments/assets/8c585d13-1221-4d2e-baa5-dd03c14b195d" width="70%">
  <p>Fig.4 - TinyVGG D, Multi-Scale(s=[36, 44]) Training Log</p>
</div>

<div align="center">
  <img src="https://github.com/user-attachments/assets/35580787-8ace-4d78-8cad-7edda5c4c3b5" width="70%">
  <p>Fig.5 - TinyVGG E, Multi-Scale(s=[36, 44]) Training Log</p>
</div>
