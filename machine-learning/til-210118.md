# 차원 축소에 관하여

source : [차원 축소 - PCA, 주성분분석 (1) by Excelsior-JH](https://excelsior-cjh.tistory.com/167)

summary

- 차원 축소의 필요성을 시각적으로 이해
- 데이터 특성에 따라 차원축소 접근법을 달리 해야 함을 이해



## 1. 차원의 저주

- 데이터 셋의 특성(feature) 증가 > 특성이 이루는 차원 수 증가
- 차원 수 증가 > 데이터 공간 부피 기하급수적으로 증가
- 데이터 공간 부피 증가 > 데이터 밀도 희소(sparse)



![jermwatt'sblog](https://i.loli.net/2021/01/18/MGrIZAp6CihbwFN.png)



데이터 차원 증가하면 데이터 포인트 간의 거리 증가

- sparse한 데이터로 머신러닝 알고리즘을 학습 > 모델 복잡성 증가
- 오버피팅 위험 가능성 높아짐
- 차원의 저주 해결 위해 학습 데이터셋 크기 늘리면 좋으나
- 데이터셋 크기 증가 속도 <<< 차원 증가 속도 (기하급수)
  - 참 힘든 일



## 2. 차원 축소를 위한 접근법

1. 투영(projection)
2. 매니폴드 학습(manifold learning)



### 2-1. 투영(projection)

- 실제 데이터셋에서는 모든 데이터의 특성, 즉 차원은 고르게 분포되어 있지 않다
- 학습 데이터셋은 고차원 공간에서 저차원 부분공간(subspace)에 위치한다
  - **= "고차원의 데이터 특성 중 일부 특성으로 데이터를 표현할 수 있다"**



![projection-from-3d-to-2d-subspace](https://i.loli.net/2021/01/18/w2nULf3IG8CWca6.png)



### 2-2. 매니폴드 학습(manifold learning)

매니폴드(manifold)의 정의

> 매니폴드는 다양체라고도 하며 국소적으로 유클리도 공간과 닮은 위상 공간이다. 국소적으로는 유클리드 공간과 구별할 수 없으나, 대역적으로 독특한 위상수학적 구조를 가질 수 있다(wikipedia)

예를 들어, 아래의 원 그림은 모든 점에 대해서 국소적으로 직선과 같은 구조를 갖는 1차원 매니폴드라 할 수 있다.

![1dim-manifold](https://i.loli.net/2021/01/18/TPkhQwYFp1E5bqo.png)



아래 그림은 스위스 롤(swiss roll, 롤케이크 모양의 데이터셋) 데이터셋이며 2D-manifold의 한 예

![swiss-roll](https://i.loli.net/2021/01/18/8aeUEfMstRjKW67.png)

* 스위스 롤 2D-manifold는 고차원(3차원) 공간에서 휘거나 말린 2D 모양
* 일반적으로 d-dim manifold는 국소적으로 d-dim 초평면으로 볼 수 있는 n차원 공간의 일부이다 (d < n)
* 스위스 롤은 d = 2, n = 3인, 국소적으로는 2D 평면이지만 3차원으로 말려있는 데이터이다.



대부분의 차원 축소 알고리즘들은...

* 이러한 manifold를 모델링하는 방식으로 동작하며, 이를 매니폴드 학습(manifold learning)이라고 한다
* 매니폴드 학습은 매니폴드 가정(manifold assumption) 또는 매니폴드 가설(manifold assumption)에 의해, 고차원 실제 데이터셋이 더 낮은 저차원 매니폴드에 가깝게 놓여 있다고 가정한다
  * 종종 다른 가정과 함께 쓰이기도
  * ex. 분류/회귀 작업 전, 학습 데이터셋을 저차원 매니폴드 공간으로 표현하면 더 간단히 문제를 해결할 수 있다는 가정

![swiss-roll-decision-boundary](https://i.loli.net/2021/01/18/nXkFECDs1jmztxO.png)



물론 이러한 가정이 항상 통하지는 않는다

* 저차원 매니폴드가 오히려 결정 경계(decision boundary)를 찾는 것이 더 어려워질 수도 있다
* **따라서 모델 학습 전, 학습 데이터셋의 차원을 감소시키면 학습 속도 향상은 보장되지만 성능 향상은 보장되지 않는다**
  * 순전히 이는 데이터셋이 어떤 모양을 갖고 있는가에 따라 달라진다

![hard-to-find-decision-boundary](https://i.loli.net/2021/01/18/ivPX86pYwjOrZ7o.png)



## PCA 맛보기



### 3.1 분산 보존

- 저차원 초평면에 데이터 투영(projection)하기 전에 적절한 초평면 찾기
- PCA는 데이터 분산이 최대가 되는 축을 찾는다
  - 원본 데이터셋과 투영 데이터셋 간 평균제곱거리 최소화하는 축
  - "분산을 최대한 보존시킨다"

![pca-preserve-maximum-variance](https://i.loli.net/2021/01/18/net9MFfb2cNiq3g.png)



### 주성분(principal component)

PCA steps

1. 학습 데이터셋에서 분산 최대 축(axis) 찾기
2. 찾은 첫번째 축과 직교(orthogonal)하면서 분산이 최대인 두번째 축 찾기
3. 첫번째, 두번째 축과 직교하고 분산 최대 보존하는 세번째 축 찾기
4. 1-3 반복하여 데이터셋 차원(특성 숫자) 수만큼의 새 축을 찾기

![pca-finds-new-axes](https://i.loli.net/2021/01/18/WlQHX9qYCjBfJDw.gif)

***이렇게 i번째 축을 정의하는 단위 벡터(unit vector)를 i-th principal component(i번째 주성분)이라고 한다

***예를 들어, 위 그림에서는 2차원 데이터셋이므로 PCA는 분산을 최대 보존하는 단위벡터 $c~1~$이 구성하는 축과 이 축에 직교하는 $c~2~$가 구성하는 축을 찾게 됨

