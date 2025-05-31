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

### #9(Rhoicosphenia abbreviata)가 전체 16개 중 6개를 배경으로 인식

<p align="center">
 <img src="https://github.com/user-attachments/assets/3f728bbc-03ac-4557-8ad7-4c5fbe5080db" width="45%"/>
 <img src="https://github.com/user-attachments/assets/3c65abc6-64fd-446c-880d-a2020d0242d2" width="45%"/>
</p>

<p align="center">
 <img src="https://github.com/user-attachments/assets/3e9d9c1c-9940-4081-baae-bd1c926eb01f" width="45%"/>
 <img src="https://github.com/user-attachments/assets/b46edaf3-7b44-4485-b0df-cb3699dce622" width="45%"/>
</p>

## RetinaNet Error cases
