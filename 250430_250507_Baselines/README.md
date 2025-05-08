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

## Reflections
