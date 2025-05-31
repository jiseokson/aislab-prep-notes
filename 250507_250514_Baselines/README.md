## Overview

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
