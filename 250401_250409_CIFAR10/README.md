## Overview
<div align="center">
  <img src="https://github.com/user-attachments/assets/9ed64963-c86c-4e6b-b02b-243771fe21bc" width="50%">
  <p>Fig.1 - TinyVGGCIFAR10의 Random sample 예측</p>
</div>

CIFAR-10 데이터셋을 대상으로 정확도 80% 이상을 달성하는 분류 모델을 설계하고 학습했다.
VGG 구조를 참고해 작은 크기의 이미지에 적합하도록 모델의 깊이와 파라미터 수를 조정하여 경량화된 구조로 구성했다.

- **PyTorch_Doc_Quickstart_250403.ipynb**\
  [PyTorch Doc Quickstart](https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html)를 보고 따라한 노트북
- **TinyVGGCIFAR10_230401.ipynb**\
  CIFAR-10을 정확도 80% 이상으로 분류하는 모델 작성 - 1 (‼️실패)
- **TinyVGGCIFAR10_230403.ipynb**\
  CIFAR-10을 정확도 80% 이상으로 분류하는 모델 작성 - 2 (✅성공)
- **TinyVGGCIFAR10_v4_chk0.pth**\
  검증 데이터셋 정확도 80% 달성한 모델 파라미터

## What I Learned
모델 설계와 실험 과정에서 측정 지표(Loss, Accuracy 등)를 기반으로 학습 상태를 분석하고 이에 맞게 대응하는 실습을 진행했다.
오버피팅과 언더피팅 패턴을 파악하는 법을 익혔고 훈련/검증 데이터셋의 Loss 및 Accuracy 변화를 근거로 모델의 상태를 판단하는 방법을 배웠다.
Loss 값이 점차 수렴하는 양상을 통해 학습이 안정적으로 이루어지고 있음을 판단할 수 있다는 점도 확인했다.

언더피팅 상황에서는 모델의 표현력을 높이기 위해 구조를 확장하고, 오버피팅일 경우 구조를 축소하거나 정규화 기법을 적용하는 등 상황별 대응 전략을 실습을 통해 익혔다.
또한 네트워크 깊이가 증가하면서 학습이 진행되지 않고 Loss가 감소하지 않는 현상(Gradient Vanishing 문제)을 경험했으며 이를 해결하기 위해 Batch Normalization을 도입했다.
그 결과 학습 안정성과 정확도가 모두 향상되는 것을 확인했다.

데이터 증강(Data Augmentation)이 특히 작은 모델의 성능에 큰 영향을 줄 수 있다는 점을 배웠고 다양한 증강 기법을 적용하는 방법도 함께 익혔다.

또한 사소하지만 Python의 실행 방식에 대해 새롭게 알게 된 점이 있다.
함수의 기본 인자는 함수가 정의되는 시점에 한 번만 평가되며, 이후 전역 객체의 이름에 다른 객체를 대입하더라도 해당 함수는 여전히 정의 시점의 객체를 기본 인자로 사용한다는 점이다.

## Code Overview
`VGGBlock(nn.Module)`과 `VGGBNBlock(nn.Module)`(Batch Normalization 적용)를 정의해 VGG 구조의 공통 모듈을 추출했다.
각 블록은 Convolution layer와 ReLU layer의 반복, 그리고 마지막에 MaxPool layer로 구성되어 있다.

공통 구조를 모듈화함으로써 모델 아키텍처를 보다 쉽게 확장할 수 있었고, 다양한 실험을 더 효율적으로 진행할 수 있었다.

```Python
class VGGBNBlock(nn.Module):
  def __init__(self, in_channels, out_channels, n_convs):
    super().__init__()

    layers = []

    for _ in range(n_convs):
      layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
      layers.append(nn.BatchNorm2d(out_channels))
      layers.append(nn.ReLU(inplace=True))
      in_channels = out_channels

    layers.append(nn.MaxPool2d(kernel_size=2))

    self.block = nn.Sequential(*layers)

  def forward(self, x):
    return self.block(x)
```

최종적으로 아래와 같은 구조의 모델을 설계하여 CIFAR-10 분류에서 정확도 80% 이상을 달성했다.
Gradient Vanishing 문제에 대응하기 위해 Batch Normalization이 적용된 블록(VGGBNBlock)을 도입했다.

```python
class TinyVGGCIFAR10_v4(nn.Module):
  def __init__(self):
    super().__init__()

    self.features = nn.Sequential(
      VGGBNBlock(3, 16, n_convs=2),
      VGGBNBlock(16, 32, n_convs=2),
      VGGBNBlock(32, 64, n_convs=2),
      VGGBNBlock(64, 128, n_convs=2),
      VGGBNBlock(128, 256, n_convs=2),
    )

    self.classifier = nn.Sequential(
      nn.Flatten(),
      nn.Linear(256, 256),
      nn.ReLU(inplace=True),
      nn.Dropout(p=0.5),
      nn.Linear(256, 256),
      nn.ReLU(inplace=True),
      nn.Dropout(p=0.5),
      nn.Linear(256, 10),
    )

  def forward(self, x):
    out = self.features(x)
    out = self.classifier(out)
    return out
```

## Reflections
모델을 설계하고 실험해나가는 과정에서 각 증상과 상황을 분석하고 이에 대응하는 실습을 직접 경험해봤다.
그동안 막연히 여겼던 작업들을 실제로 해보며 이전에 배웠던 개념들을 구체적인 상황에 적용해볼 수 있어 의미가 있었다.

다양한 구조를 설계하고 실험하는 과정이 흥미로웠으나 그 과정이 중구난방으로 느껴지기도 했다.
체계적이고 과학적인 실험 방법론을 익혀보고 싶고 이를 기반으로 다른 모델도 설계해보고 싶다.

## References
- [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)
- [CNN Explainer](https://poloclub.github.io/cnn-explainer/)
