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
