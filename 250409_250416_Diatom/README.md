## Overview
<div align="center">
  <img src="https://github.com/user-attachments/assets/86d4020b-0ecf-423d-bb3d-bbc1baf21b69" width="75%">
  <p>Fig.1 - Diatom Benchmark Dataset의 Random samples</p>
</div>

Diatom Benchmark Dataset의 기본적인 특성을 조사했다.

## Class Distribution Analysis
데이터의 불균형이 매우 심하다. 상위 2~3개의 클래스(Gomphonema olivaceum, Navicula cryptotenella 등)가 샘플의 대다수를 차지하고 있다.

<div align="center">
  <img src="https://github.com/user-attachments/assets/9246c4da-3ce3-411f-b036-bd1cf2e501d3" width="80%">
  <p>Fig.2 - Class 분포</p>
</div>

## Objects per Image
대부분 1~2개의 객체가 존재하며 평균은 1.38로 낮은 수준이다.

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
