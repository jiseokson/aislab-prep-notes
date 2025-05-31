# [CIFAR10](https://github.com/jiseokson/aislab-prep-notes/tree/main/250401_250409_CIFAR10)

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

# [TinyVGG](https://github.com/jiseokson/aislab-prep-notes/tree/main/250401_250409_VGG)

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
이는 CIFAR-10과 ImageNet 간의 이미지 해상도 비율인 1/7(=32/224)와 유사하다.
본 실험에서는 이러한 분석을 바탕으로 논문에 제시된 VGG 구조에서 Convolution 레이어의 채널 수를 1/8로 축소한 모델을 이용했다.

<div align="center">
  <table border="1" cellspacing="0" cellpadding="5">
    <caption><b>Table.1 - TinyVGG configuration</b></caption>
    <thead>
      <tr>
        <th align="center">A</th>
        <th align="center">B</th>
        <th align="center">C</th>
        <th align="center">D</th>
        <th align="center">E</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td align="center">conv3-8</td>
        <td align="center">conv3-8<br>conv3-8</td>
        <td align="center">conv3-8<br>conv3-8</td>
        <td align="center">conv3-8<br>conv3-8</td>
        <td align="center">conv3-8<br>conv3-8</td>
      </tr>
      <tr>
        <td align="center" colspan="5">maxpool2</td>
      </tr>
      <tr>
        <td align="center">conv3-16</td>
        <td align="center">conv3-16<br>conv3-16</td>
        <td align="center">conv3-16<br>conv3-16</td>
        <td align="center">conv3-16<br>conv3-16</td>
        <td align="center">conv3-16<br>conv3-16</td>
      </tr>
      <tr>
        <td align="center" colspan="5">maxpool2</td>
      </tr>
      <tr>
        <td align="center">conv3-32<br>conv3-32</td>
        <td align="center">conv3-32<br>conv3-32</td>
        <td align="center">conv3-32<br>conv3-32<br>conv1-32</td>
        <td align="center">conv3-32<br>conv3-32<br>conv3-32</td>
        <td align="center">conv3-32<br>conv3-32<br>conv3-32<br>conv3-32</td>
      </tr>
      <tr>
        <td align="center" colspan="5">maxpool2</td>
      </tr>
      <tr>
        <td align="center">conv3-64<br>conv3-64</td>
        <td align="center">conv3-64<br>conv3-64</td>
        <td align="center">conv3-64<br>conv3-64<br>conv1-64</td>
        <td align="center">conv3-64<br>conv3-64<br>conv3-64</td>
        <td align="center">conv3-64<br>conv3-64<br>conv3-64<br>conv3-64</td>
      </tr>
      <tr>
        <td align="center" colspan="5">maxpool2</td>
      </tr>
      <tr>
        <td align="center">conv3-64<br>conv3-64</td>
        <td align="center">conv3-64<br>conv3-64</td>
        <td align="center">conv3-64<br>conv3-64<br>conv1-64</td>
        <td align="center">conv3-64<br>conv3-64<br>conv3-64</td>
        <td align="center">conv3-64<br>conv3-64<br>conv3-64<br>conv3-64</td>
      </tr>
      <tr>
        <td align="center" colspan="5">maxpool2</td>
      </tr>
      <tr>
        <td align="center" colspan="5">FC-64</td>
      </tr>
      <tr>
        <td align="center" colspan="5">FC-64</td>
      </tr>
      <tr>
        <td align="center" colspan="5">FC-10</td>
      </tr>
      <tr>
        <td align="center" colspan="5">soft-max</td>
      </tr>
    </tbody>
  </table>
</div>

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

## Code Overview

### `PairedFlipDataset` and `MultiTransformDataset`
논문에서 제시된 다양한 증강 기법(Augmentation)과 평가 기법을 적용하기 위해 커스텀 데이터셋을 정의하여 활용하였다.

`PairedFlipDataset`은 기본 데이터셋으로부터 원본 이미지와 그 수평 반전 이미지를 함께 반환하는 구조로,
각 이미지에 대해 Softmax 출력을 계산한 후 평균을 내는 Testing 방식에 사용된다.
```python
class PairedFlipDataset(Dataset):
  def __init__(self, base_dataset):
    self.base_dataset = base_dataset
    
  def __len__(self):
    return len(self.base_dataset)
  
  def __getitem__(self, idx):
    img, label = self.base_dataset[idx]
    
    img_flipped = T.hflip(img)
    
    return img, img_flipped, label
```

`MultiTransformDataset`은 하나의 샘플에 여러 Transform을 동시에 적용한 결과를 반환하도록 설계되었으며,
이는 Multi-Scale Evaluation을 구현하기 위한 용도로 활용된다.
```python
class MultiTransformDataset(Dataset):
  def __init__(self, base_dataset, trans):
    self.base_dataset = base_dataset
    self.trans = trans
    
  def __len__(self):
    return len(self.base_dataset)
  
  def __getitem__(self, idx):
    *imgs, label = self.base_dataset[idx]
    
    imgs = [tran(img) for img in imgs for tran in self.trans]
    
    return *imgs, label
```


### `Trainer`
모델 학습에 필요한 손실 함수(Loss), 옵티마이저(Optimizer), 모델 정의, 로그 관리 등의 요소를 일관성 있게 관리하기 위해 학습 파이프라인을 Trainer 클래스로 구성하였다.

```python
trainer = Trainer(
  TinyVGG_A,
  train_transform=fix_scale_transform(s=36),
  test_transforms=[
    fix_scale_transform(s=36),
    fix_scale_transform(s=40)
  ]
)

trainer.train(epoch=10) # Train for the first 10 epochs
trainer.train(epoch=5) # Continue training for 5 more epochs

trainer.evaluate() # Evaluate the model on the validation set
```

## Results
TinyVGG 구조 A부터 E까지에 대해 다양한 학습 이미지 스케일 설정을 실험한 결과,
전반적으로 구조가 깊어질수록 성능은 향상되었지만 학습 전략에 따라 그 효과가 달라지는 양상을 보였다.

$S=36$에서 학습한 경우, 가장 단순한 구조인 A는 `0.783`의 정확도를 기록했으며, B는 그보다 약간 높은 `0.798`의 정확도를 달성하였다.
구조가 더 복잡한 C, D, E는 오히려 낮은 성능을 보였는데, 이는 모델 구조에 비해 데이터의 다양성이 복잡했기 때문으로 해석된다.

Multi-Scale 학습($[36, 44]$)을 적용한 경우, D와 E에서 각각 `0.785`, `0.732`로 정확도가 상승하는 모습을 보였다.
특히 D는 실험 전체에서 가장 높은 정확도를 기록했다.
이는 다양한 해상도의 이미지를 활용한 학습이 일반화 성능 향상에 효과적임을 보여준다.

단순히 구조를 깊게 하는 것보다는 학습 데이터에 적절한 다양성을 제공하는 것이 더 효과적인 성능 향상 방법일수도 있다.

<div align="center">
  <table border="1" cellspacing="0" cellpadding="5">
    <caption><b>Table.2 - TinyVGG performance at multiple scales</b></caption>
    <thead>
      <tr>
        <th rowspan="2">TinyVGG config</th>
        <th>Train Size (S)</th>
        <th>Test Size (Q)</th>
        <th>Accuracy</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td rowspan="1">A</td>
        <td>36</td>
        <td>36</td>
        <td>0.783</td>
      </tr>
      <tr>
        <td rowspan="1">B</td>
        <td>36</td>
        <td>36</td>
        <td>0.798</td>
      </tr>
      <tr>
        <td rowspan="3">C</td>
        <td>36</td>
        <td>36</td>
        <td>0.771</td>
      </tr>
      <tr>
        <td>40</td>
        <td>40</td>
        <td>0.756</td>
      </tr>
      <tr>
        <td>[36, 44]</td>
        <td>40</td>
        <td>0.753</td>
      </tr>
      <tr>
        <td rowspan="3">D</td>
        <td>36</td>
        <td>36</td>
        <td>0.774</td>
      </tr>
      <tr>
        <td>40</td>
        <td>40</td>
        <td>0.761</td>
      </tr>
      <tr>
        <td>[36, 44]</td>
        <td>40</td>
        <td>0.785</td>
      </tr>
      <tr>
        <td rowspan="3">E</td>
        <td>36</td>
        <td>36</td>
        <td>0.679</td>
      </tr>
      <tr>
        <td>40</td>
        <td>40</td>
        <td>0.729</td>
      </tr>
      <tr>
        <td>[36, 44]</td>
        <td>40</td>
        <td>0.732</td>
      </tr>
    </tbody>
  </table>
</div>

## What I learned
데이터 증강 기법(Augmentation)이 모델 성능에 큰 영향을 미친다는 사실을 실험을 통해 체감할 수 있었다.
논문에서 제시된 증강 기법을 재현하기 위해 PyTorch의 다양한 `transforms` 함수를 찾아보고 직접 테스트해보며 사용법을 익혔다.

모델 학습이 완료된 이후에도 다양한 평가 기법이 존재함을 새롭게 알게 되었다.
논문에서는 주로 Dense Evaluation 방식을 통해 모델을 평가하지만
본 실험에서는 CIFAR-10의 작은 이미지 크기 특성상 해당 기법을 그대로 적용하기 어려웠다.
다만 이후의 다른 실습에서는 Dense Evaluation을 직접 구현해보고 싶다.

이러한 다양한 증강 및 평가 기법을 실험에 적용하기 위해
여러 `transform`을 동시에 적용할 수 있는 `MultiTransformDataset`과
수평 반전된 이미지 쌍을 함께 처리할 수 있는 `PairedFlipDataset` 등 커스텀 `Dataset`도 직접 구현해보았다.
이 과정 역시 실험 재현과 정확한 평가에 있어 중요한 역할을 했다.

## Reflections
논문에 제시된 실험 과정을 가능한 한 충실히 재현하는 것을 이번 실험의 주요 목표로 삼았다.
모델을 직접 구현해본 경험은 있었지만, 실험을 계획하고 실행하며 결과를 분석해본 경험은 이번이 처음이었다.
그 과정을 통해 실험과 결과 해석이 하나의 주장을 뒷받침하는 데 얼마나 중요한 역할을 하는지를 실제로 체감할 수 있었다.

특히 실험 결과를 해석하고 정리하는 과정이 가장 어려웠다.
다음에는 분석 기준을 미리 설정하고 결과를 보다 체계적으로 정리하는 방법을 시도해보고자 한다.

또한 훈련 로그를 확인하며 추가 학습 여부를 판단하는 과정이 꽤 번거롭게 느껴졌다.
실험을 효율화하기 위해 다양한 시도를 해보았지만 훈련 반복 횟수(epoch)를 결정하는 기준에 대해서는 깊이 있게 고민하지 못했다.
앞으로는 훈련 종료 조건이나 반복 횟수를 더 합리적으로 설정하는 방법을 학습하고,
실험을 더욱 편리하게 수행할 수 있도록 전체 파이프라인을 리팩터링하는 것도 시도해보고 싶다.

## References
- [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)

# [Diatom](https://github.com/jiseokson/aislab-prep-notes/tree/main/250409_250416_Diatom)

## Overview
<div align="center">
  <img src="https://github.com/user-attachments/assets/86d4020b-0ecf-423d-bb3d-bbc1baf21b69" width="75%">
  <p>Fig.1 - Diatom Benchmark Dataset의 Random samples</p>
</div>

[Diatom Benchmark Dataset](https://www.kaggle.com/datasets/huseyingunduz/diatom-dataset)의 기본적인 특성과 분포를 조사했다.

규조류(Diatoms)는 바다, 민물, 습한 토양 및 다양한 표면에서 발견되는 조류(algae)의 일종으로,
자연에서 가장 흔하게 발견되는 식물성 플랑크톤(phytoplankton) 중 하나이다.
현재까지 200개 이상의 속(genera)과 약 20만 종(species)이 존재하는 것으로 알려져 있다.
규조류는 지구 전체 산소의 약 20~25%를 생산할 정도로 중요한 생물군이다.
\[Source: [Kaggle, Diatom Dataset](https://www.kaggle.com/datasets/huseyingunduz/diatom-dataset)\]

해당 데이터셋은 2,197장의 컬러 이미지로 이루어져 있으며,
68종의 규조류(diatom)에 해당하는 총 3,027개의 객체가 어노테이션 되어있다.
모든 이미지의 해상도는 2112×1584 픽셀이다.

## Class Distribution Analysis
데이터의 불균형이 매우 심하다. 상위 2~3개의 클래스(Gomphonema olivaceum, Navicula cryptotenella 등)가 샘플의 대다수를 차지하고 있다.

<div align="center">
  <img src="https://github.com/user-attachments/assets/9246c4da-3ce3-411f-b036-bd1cf2e501d3" width="80%">
  <p>Fig.2 - Class 분포</p>
</div>

## Objects per Image
대부분 1~2개의 객체가 존재하며 평균은 1.38로 낮은 수준이다.
바운딩 박스가 존재하지 않는 이미지 샘플이 발견되었다. 훈련시 제거가 필요해보인다.

Single-Object Detection 환경에 가깝다.

<div align="center">
  <img src="https://github.com/user-attachments/assets/25a4c566-8d9b-4173-a7bc-716cceec0c93" width="60%">
  <p>Fig.3 - #Objects/Image 분포</p>
</div>

## Bounding Box Size Analysis
Bounding Box의 면적이 좌로 치우쳤다(right-skewed). 즉 크기가 작은 객체가 많다.
중앙값과 평균값의 차이가 크며 극단적인 outlier들이 다수 존재한다.

Small-Object Detection 환경으로 볼 수 있다.

<div align="center">
  <img src="https://github.com/user-attachments/assets/1f5738a4-7dce-40d6-82e8-7bf4a2f9ee36" width="60%">
  <p>Fig.4 - Bounding Box 면적 분포</p>
</div>

## Aspect Ratio Analysis
대부분 1에 가까우며 정사각형 또는 가로가 약간 긴 직사각형 형태이다.

<div align="center">
  <img src="https://github.com/user-attachments/assets/15af86b6-2cb3-4385-8a38-72ae1d54266d" width="60%">
  <p>Fig.5 - Aspect Ratio 분포</p>
</div>

## Spatial Distribution of Objects
객체가 이미지 전체에 걸쳐 넓게 분포하고 있으며, 특히 중앙의 밀도가 높다.

<div align="center">
  <img src="https://github.com/user-attachments/assets/0a12b38a-204b-49e7-8350-3f3174c1def0" width="60%">
  <p>Fig.6 - 객체의 공간 분포</p>
</div>

## What I Learned
raw dataset으로부터 이미지와 어노테이션 데이터를 직접 추출하고 분석해보는 실습을 진행했다.
이 과정에서 PASCAL VOC 형태의 어노테이션뿐만 아니라 다양한 형식의 파일 구조(MS COCO 등)에 대해 새롭게 알게 되었다.

시각화를 더 편하게 하기 위해 클래스 수를 기준으로 내림차순 정렬하여 클래스 인덱스를 새롭게 정의해보았다.
plt를 이용해 히스토그램, 바 플롯, 산점도 등 다양한 그래프를 직접 그려보면서 그동안 애매하게만 알고 넘어갔던 옵션들을 문서와 함께 꼼꼼히 확인할 수 있었다.
또한 데이터셋에 대해 일반적으로 어떤 분석이 수행되는지 감을 잡았고, 클래스 분포나 바운딩 박스의 크기·비율 같은 요소들이 모델 학습에 어떤 영향을 줄 수 있는지도 함께 배웠다.

## Reflections
이미지, 클래스, 바운딩 박스 등 서로 연결된 정보를 일관성 있게 관리하는 작업이 생각보다 훨씬 까다로웠다.
각 요소들 간의 매핑을 어떻게 정의하고 어떤 방식으로 정보를 기록해둘지를 많이 고민하게 됐다.

PyTorch를 써왔던 경험이 분석 코드를 구성할 때 적잖은 도움이 되었고, 이런 파일 포맷 처리나 구조화 경험이 이후 Baseline 모델을 구성하고 학습하는 데 발판이 되면 좋겠다.

# [Baseline 1](https://github.com/jiseokson/aislab-prep-notes/tree/main/250430_250507_Baselines)

## Overview

Diatom Dataset에 대해 대표적인 Object Detection 모델들을 적용하여 성능 베이스라인을 확보하는 것을 목표로 한다.
주요 실험 모델로는 Faster R-CNN과 RetinaNet이 사용되었으며 각각에 대해 head만 학습하는 방식, 전체 fine-tuning, 그리고 다중 learning rate 적용 등 다양한 설정을 적용해 비교하였다.

- 데이터: 68종의 규조류(Diatom)가 포함된 2,197장의 이미지

- 평가 지표: `mAP@[.5:.95]`, `mAP@50`, `mAP@75`

- 목적: 이후 모델 개선/최적화 실험의 기준선(Baseline) 확보

## Code Overview

### `Trainer`

학습 파이프라인의 일관된 관리를 위해 Model, Optimizer, Dataloader를 통합적으로 구성하고 제어할 수 있는 관리 툴을 제작했다.

```python
>>> trainer = Trainer(model, optimizer, train_dataloader, test_dataloader)

>>> trainer.train(epoch=5, checkout=True) # 첫 5 에포크 학습
    # ... 로그, 지표, 그래프 출력

>>> trainer.train(epoch=5, checkout=True) # 추가 5 에포크 학습
    # ... 로그, 지표, 그래프 출력

>>> trainer.load_checkpoint(checkpoint=5) # 해당 체크포인트로 모델, 옵티마이저 이동

>>> trainer.evaluate()
    # ... 지표, 그래프 출력
```

### `get_fasterrcnn_model(num_classes)`
Pre-trained 된 Faster R-CNN 모델을 로드한다

```python
def get_fasterrcnn_model(num_classes):
  weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
  model = fasterrcnn_resnet50_fpn(weights=weights)

  in_features = model.roi_heads.box_predictor.cls_score.in_features
  model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

  return model
```

### `get_retinanet_model(num_classes)`

Pre-trained 된 RetinaNet 모델을 로드한다

```python
def get_retinanet_model(num_classes):
  weights = RetinaNet_ResNet50_FPN_Weights.DEFAULT
  model = retinanet_resnet50_fpn(weights=weights)

  in_features = model.backbone.out_channels
  num_anchors = model.head.classification_head.num_anchors

  model.head.classification_head = RetinaNetClassificationHead(
    in_channels=in_features,
    num_anchors=num_anchors,
    num_classes=num_classes
  )

  return model
```

## Faster R-CNN (ResNet-50 FPN) – Head Finetuning (Epoch 20)

![image](https://github.com/user-attachments/assets/ea29c513-b4c3-472a-8ddb-15343429342b)

## Faster R-CNN (ResNet-50 FPN) – Full Finetuning (Epoch 20)

![image](https://github.com/user-attachments/assets/dd1e1b6e-ad2b-45bf-bb55-dea8f0048049)

## RetinaNet (ResNet-50 FPN) – Head Finetuning (Epoch 20)

![image](https://github.com/user-attachments/assets/61283869-6184-4fea-80d5-f7877f154366)

## RetinaNet (ResNet-50 FPN) – Full Finetuning (Epoch 25)

![image](https://github.com/user-attachments/assets/7090c560-0464-4d26-b3f4-5dbfa51ca244)

## RetinaNet (ResNet-50 FPN) – Backbone (lr=1e-4), Head (lr=1e-2) (Epoch 20)

![image](https://github.com/user-attachments/assets/c55a974b-140e-439c-87f7-6574a20bcd70)

## What I Learned

PyTorch 기반 Object Detection 모델의 기본적인 사용법을 익혔다.
모델의 모드(train(), eval())에 따라 forward 함수의 출력이 달라진다는 점을 이해했고, 각 출력값의 의미를 파악하여 학습 및 평가 루프 구현에 직접 적용해보았다.

또한, Object Detection에서 널리 사용되는 평가 지표인 mAP(mean Average Precision)의 개념을 학습하였다.
특히 IoU(Intersection over Union) 기준에 따라 다양한 세부 지표(`mAP@[.5:.95]`, `mAP@50`, `mAP@75` 등)로 나뉘며 각 지표가 가지는 의미와 활용 목적에 대해서도 이해할 수 있었다.

교수님의 리뷰를 바탕으로 학습 로그 그래프를 분석하며 모델의 학습 상태를 점검해보았다.
이를 통해 성능 변화의 원인을 파악하고 이후 실험 방향에 대해서도 일부 계획을 수립할 수 있었다.

## Reflections
교수님의 리뷰 결과 현재 학습 로그를 기준으로는 모델이 아직 충분히 수렴하지 않은 것으로 보인다. 이에 따라 동일한 설정으로 학습을 추가 진행할 예정이다.

# [Baseline 2](https://github.com/jiseokson/aislab-prep-notes/tree/main/250507_250514_Baselines)

## Overview

![image](https://github.com/user-attachments/assets/accde4af-ce28-4a6d-a325-79734b6e5ae2)

![image](https://github.com/user-attachments/assets/664d22bd-4448-486a-b93d-e530bae4f97b)

학습 파이프라인을 일관되게 관리하고 언제든지 재학습이 가능하도록 모델의 가중치, 옵티마이저 상태 등을 통합적으로 구성하고 제어할 수 있는 프로그램을 작성했습니다.
- 모델 가중치, 옵티마이저 상태, 평가 지표를 주기적으로 체크포인트로 저장
- 최신 체크포인트를 로드하여 학습을 중단한 지점부터 재시작 가능
- Weights & Biases(W&B)를 통한 실시간 로깅 및 모니터링 지원
- 학습 완료 후, 모든 체크포인트에 대해 평가 지표를 일괄 측정

## How To Run

### 1. Train

학습에 앞서 주요 하이퍼파라미터를 YAML 파일로 설정합니다:

```yaml
# Faster R-CNN 예시

model: fasterrcnn
mode: full

learning_rate: 0.005
```

```yaml
# RetinaNet 예시

model: retinanet
mode: split

back_lr: 0.0001
head_lr: 0.01
```

- `model`은 사용할 모델 이름이며, `fasterrcnn` 또는 `retinanet`을 지정할 수 있습니다.
- `mode`는 fine-tuning 전략을 나타내며 `full`, `head`, `split` 중 하나를 선택합니다.
- `mode=full` 또는 `mode=head`인 경우 `learning_rate`를 사용하고, `mode=split`일 경우 `back_lr과` `head_lr`를 각각 설정해야 합니다.

하이퍼파라미터 구성이 완료되면 아래 명령어로 학습을 시작할 수 있습니다:

```bash
python train.py --config CONFIG_PATH --epoch 20
```

추가 학습이 필요한 경우 다음과 같이 이어서 학습할 수 있습니다:

```bash
python train.py --config CONFIG_PATH --epoch 5
```

### 2. After Train Evaluation

학습이 완료된 후 각 에포크의 체크포인트에 대해 학습 과정 중 측정되지 않았던 지표들을 추가로 측정합니다:

```bash
python measure.py --config CONFIG_PATH
```

## What I Learned

이전 연구에서는 매번 모델을 새로 학습하고 불러오는 과정이 번거로웠다.
Jupyter Notebook은 변수들을 빠르게 수정하며 실험하기에는 편리했지만, 장시간이 소요되는 학습을 반복적으로 수행하기에는 다소 불안정했다.
특히 세션이 끊기면 학습도 함께 종료되기 때문에 학습 중 내내 세션이 유지되도록 확인해야 하는 점이 비효율적으로 느껴졌다.

이러한 문제를 해결하기 위해 학습에 관련된 코드를 모두 `.py` 파일로 분리하고 실험마다 달라지는 하이퍼파라미터를 외부에서 손쉽게 설정할 수 있도록 구성된 프로그램을 작성했다.
이제는 `tmux` 세션을 통해 학습 스크립트를 실행한 후 `SSH` 연결을 끊더라도 백그라운드에서 안전하게 학습이 계속 진행된다.
또한 W&B 로깅을 통합하여 실시간으로 학습 추이를 모니터링할 수 있게 되면서 실험의 신뢰성과 효율이 크게 향상되었다.

## Reflections

학습 파이프라인을 체계적으로 관리하는 것이 얼마나 중요한지 실감할 수 있었다.
단순히 사용 편의성을 높이는 것을 넘어 모델 가중치와 옵티마이저 상태 등 다양한 정보를 일관되게 관리할 수 있다는 점이 큰 강점으로 다가왔다.
아직 배우는 단계이지만 전체적인 머신러닝 워크플로우에 대한 큰 그림이 점점 그려지기 시작하는 느낌이다.

# [Confusion Matrix](https://github.com/jiseokson/aislab-prep-notes/tree/main/250521_250528_Baselines)

## Overview

학습된 모델의 객체 인식 능력에 대해 조사했다. Confusion matrix 시각화와 실제 예측 결과를 분석함으로써 현재 시스템의 문제점을 파악했다.

## Validation Dataset Class Distribution

Validation Dataset의 class 분포를 확인했다.
![image](https://github.com/user-attachments/assets/910a4cc4-e5e6-4086-8d4e-7e864433616d)

```
 1:Gomphonema olivaceum                    : 91
 2:Navicula cryptotenella                  : 63
 3:Fragilaria recapitellata                : 37
 4:Encyonema silesiacum                    : 39
 5:Navicula reichardtiana                  : 36
 6:Planothidium lanceolatum                : 21
 7:Gomphonema tergestinum                  : 18
 8:Navicula cryptotenelloides              : 16
 9:Rhoicosphenia abbreviata                : 16
10:Meridion circulare                      : 17
.
.
.
68:Surella minuta                          : 1
```

## Confusion matrix

학습된 Faster R-CNN과 RetinaNet 모델을 선정해 Confusion matrix를 계산해 시각화하였다.
Faster R-CNN은 Full Fine-tuning의 `epoch=20`, RetinaNet은 Full Fine-tuning의 `epoch=55`를 선정해 실험했다.

<p align="center">
 <img src="https://github.com/user-attachments/assets/510942b9-3525-456a-aec4-667673a7a822" width="45%"/>
 <img src="https://github.com/user-attachments/assets/4c2c64c4-1419-40d0-ad39-f570ec600bf9" width="45%"/>
</p>

## Faster R-CNN Error cases

### #9(Rhoicosphenia abbreviata) 전체 16개 중 배경으로 6개를 인식

<p align="center">
 <img src="https://github.com/user-attachments/assets/3f728bbc-03ac-4557-8ad7-4c5fbe5080db" width="45%"/>
 <img src="https://github.com/user-attachments/assets/3c65abc6-64fd-446c-880d-a2020d0242d2" width="45%"/>
</p>

```
Ground Truth: [9]
Prediction: []
```

<p align="center">
 <img src="https://github.com/user-attachments/assets/3e9d9c1c-9940-4081-baae-bd1c926eb01f" width="45%"/>
 <img src="https://github.com/user-attachments/assets/b46edaf3-7b44-4485-b0df-cb3699dce622" width="45%"/>
</p>

```
Ground Truth: [9]
Prediction: [3, 44, 41, 5, 8, 46]
```

### #12(Encyonema ventricosum) 전체 14개 중 #4(Encyonema silesiacum)로 4개 인식

<p align="center">
 <img src="https://github.com/user-attachments/assets/be776398-3d53-4799-8b69-ac2d94722d87" width="45%"/>
 <img src="https://github.com/user-attachments/assets/dfc6d2c9-095f-4847-b05c-07d31daf6481" width="45%"/>
</p>

```
Ground Truth: [12]
Prediction: [12, 4]
```

### #17(Diatoma mesodon) 전체 6개 중 배경으로 7개, #6(Planothidium lanceolatum)로 4개를 인식

<p align="center">
 <img src="https://github.com/user-attachments/assets/3ef6bd44-44b3-4b30-8ce3-fdd0316ee34a" width="45%"/>
 <img src="https://github.com/user-attachments/assets/1deb2c9b-8451-4e01-a0d2-258572fcc5ec" width="45%"/>
</p>

```
Ground Truth: [17, 6]
Prediction: [6]
```

<p align="center">
 <img src="https://github.com/user-attachments/assets/9c697561-4788-4612-989f-5a69ae2e021f" width="45%"/>
 <img src="https://github.com/user-attachments/assets/dc088bd8-db81-405c-ae33-bedc4fb43246" width="45%"/>
</p>

```
Ground Truth: [17]
Prediction: [6]
```

## RetinaNet Error cases

<p align="center">
 <img src="https://github.com/user-attachments/assets/8c11e872-2452-428d-9227-700299e58d06" width="45%"/>
 <img src="https://github.com/user-attachments/assets/b5c0d64b-124c-4726-8d50-8f07a655b492" width="45%"/>
</p>

```
Ground Truth: [34, 27]
Prediction: [27, 34, 8, 2, 41, 34, 13, 36, 34, 39]
```

## Conclusion

예측 결과를 직접 시각화함으로써, 모델이 따르는 일반적인 분류 패턴을 확인할 수 있었다.
현재 사용 중인 데이터셋의 클래스 이름은 규조류(Diatom)의 ‘속(Genus)’과 ‘종(Species)’으로 구성되어 있다.

#12(Encyonema ventricosum) 클래스의 14개 샘플 중 4개가 #4(Encyonema silesiacum)으로 잘못 분류된 사례를 보면, 두 클래스 모두 ‘Encyonema’라는 동일한 속(Genus)에 속해 있어 혼동이 발생했음을 알 수 있다.

또한 형태가 유사한 경우 학습 데이터에서 샘플 수가 더 많은 클래스로 분류되는 경향이 있다. 실제로 #17(Diatoma mesodon)의 6개 샘플 중 4개가 #6(Planothidium lanceolatum)으로 인식된 사례에서 이러한 경향이 확인되었다.

RetinaNet에서 두드러진 관찰 결과는 박스의 위치는 비교적 정확하게 예측하지만 너무 많은 객체를 탐지하여 NMS(Non-Maximum Suppression)로도 충분히 걸러지지 않는 경우가 많았다는 점이다.
이로 인해 실제보다 과도하게 많은 객체가 검출되는 문제가 자주 발생했다.
