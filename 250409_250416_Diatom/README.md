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
