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
