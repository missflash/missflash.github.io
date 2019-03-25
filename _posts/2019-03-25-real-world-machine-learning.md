---
title: "Real World Machine Learning"
date: 2019. 3. 25. 오후 8:56:55
categories:
use_math: true
---

# 1장 머신러닝이란 무엇인가?
* 머신러닝의 다섯 가지 이점
  * 정확성
  * 자동화
  * 신속성
  * 맞춤화
  * 확장성
* 모델 성능 최적화
  * 모델 매개변수 조정
  * 특성의 부차 집합을 선택
  * 데이터 전처리


# 2장 실무현장 데이터
* 데이터 수집시 고려해야 하는 사항들
  * 어떤 특성을 포함해야 하는가?
  * 목표 변수에 대한 실측 자료를 어떻게 얻을 수 있는가?
  * 얼마나 많은 훈련 데이터가 필요한가?
  * 훈련 집합이 충분히 대표성을 띄는가?
* 데이터 전처리 방법
  * 범주형 특성 고려
  * 결측 자료 다루기
  * 간단한 특성 추출
  * 데이터 정규화
* 특성 정규화 수식
<script src="https://gist.github.com/missflash/29578dc5eeaef7bcc65e4ad9d4f35eef.js"></script>


# 3장 모델링과 예측
* $Y = f(X)$
  * $Y$ : 예측
  * $f$ : 추정
  * $X$ : 신규
* 지도 학습
  * 분류 : 목표가 범주형일 때
  * 회귀 : 목표가 수치형일 때
* 비지도 학습
* 선형 vs. 비선형


# 4장 모델 평가와 최적화
* 커널 스무딩 (Kernel Smoothing)
* 대역폭 매개변수 (Bandwidth Parameter)
  * 작으면 : 과적합
  * 크면 : 과소적합
* 신규 데이터의 예측 정확도 평가
  * 평균 제곱 오차 (Mean Squared Error)
  * 교차 검증 (Cross Validation)
  * 홀드아웃 메소드 (Holdout Method)
  * k겹 교차 검증 (k-fold Cross Validation)
* 분류 모델 평가
  * 정확도 (Accuracy)
  * 혼동 행렬 (Confusion Matrix)
    * 참 양성률 (민감도, Sensitivity) : 진짜True, 예측True
    * 참 음성률 (특이도, Specificity) : 진짜False, 예측False
    * 거짓 양성률 (폴아웃, Fall-Out) : 진짜False, 예측True
    * 거짓 음성률 (결측률, Miss Rate) :진짜True, 예측False
  * 수신자 조작 특성 (Receiver Operating Characteristics, ROC)
  * ROC 곡선 아래 영역 (Area Under Curve, AUC)
* 회귀 모델 평가
  * 제곱근 평균 제곱 오차 (Root-Mean-Square Error, RMSE)
  * 잔차 (Residual) 검사
* 매개변수 조율을 통한 모델 최적화
  * 로지스틱 회귀 : 없음
  * K 최근접 이웃 : 평균에 가장 가까운 이웃의 수
  * 의사결정트리 : 분할기준, 최대 깊이 트리, 분할에 필요한 최소 표본
  * 커널 서포트 벡터 머신 : 핵 유형, 계수, 벌점 매개변수 신청
  * 랜덤 포레스트 : 트리의 수, 특성의 수, 분할 기준, 최소 표본
  * 부스팅 : 트리의 수, 학습 속도, 트리의 최대 깊이, 최소 표본
* 그리드 탐색


# 5장 특성 추출의 기본
* 특성 추출을 해야 하는 다섯 가지 이유
  * 데이터를 변형해 목표와 관련 짓기 (Ex. 잔고 대비 대출 비율 등)
  * 외부 데이터 가져오기
  * 비정형 데이터 자료 사용
  * 해석하기 쉬운 특성 만들기
  * 많은 특성 집합을 사용해 창조성 향상
* 특성 추출 과정
  * 날짜 및 시간 특성 : 시간, 요일, 월 정보 등을 카테고리 특성으로 변경 (Ex. 00~23시, 1월~12월 등)
  * 텍스트 특성 : 단어 주머니 (Bag of Words), 정지어 (Stop Words)
* 특성 선택
  * 특성 중요도
  * 전진 선택 : 최상의 특성 집합이 선택될 때까지 특성을 반복해서 추가
  * 후진 제거 : 최악인 모델 특성을 집합에서 반복해서 제거
  * 차원 감소


# 6장 예제: 뉴욕 시 택시 데이터
* 데이터
  * 데이터 시각화
  * 머신러닝 작업에서의 두 가지 함정
    * 믿기지 않는 시나리오 (Too-Good-To-Be-True Scenario)
    * 섣부른 가정 (Premature Assumptions)
* 모델링
  * 선형 모델
  * 비선형 분류기 (Ex. 랜덤 포레스트 등)


# 7장 고급 특성 추출 기법
* 고급 텍스트 특성
  * 단어 주머니 모델
    * 토큰화 : 텍스트를 조각들로 분해하는 일
      * n-gram : n개 글자로 구성된 문서
      * unigram, bigram, trigram...
    * 변형 : fox vs. Fox
    * 어간 추출 : jump, jumping, jumps, ...
    * 벡터화 (Vectorization)
    * 정지어 (Stop Words)
  * 주제 모델링
    * 용어 빈도-역문서 빈도 (Term Frequency-Inverse Document Frequency, TF-IDF)<br>
$tf-idf(용어, 문서, 문서들)=개수(문서내 용어)\frac{개수(문서들)}{개수(용어가 들어있는 문서들)}$
    * 잠재 의미 분석 (Latent Sementic Analysis, LSA)
      * T : 용어(개집, 짖는 소리)-개념(개) 행렬
      * D : 개념-문서 행렬
      * S : 특이값 행렬
    * 확률론적 방법 (Latent Dirichlet Analysis, LDA)
  * 내용 확장
    * 링크
    * 지식베이스 확장
    * 텍스트 메타 특성
* 이미지 특성
  * 간단한 이미지 특성
    * 색상
    * 메타데이터
  * 물체와 형태 추출
    * 윤곽선 검출 (Edge Detection)<br>
$EdgeScore=\frac{\Sigma(edges)}{res_x*res_y}$
    * 고급 형태 특성 (HOG)
<script src="https://gist.github.com/missflash/e040bf8ae14abec32525e8e87c01aabc.js"></script>
    * 차원 감소 (PCA, Diffusion Map)
    * 자동 특성 추출 (DNN, DBN)
* 시계열 특성
  * 시계열 데이터의 유형
    * 고전적 시계열 : 시간의 흐름에 맞춰 수치를 측정한 값으로 구성
    * 점 과정 (Point Processes) : 시간이 지나면서 때때로 발생하는 사건을 모은 것
  * 시계열 데이터를 바탕으로 한 예측
    * 시계열 예측 : 단일 시계열 대상
    * 시계열 분류 or 시계열 회귀 : 다수의 시계열 대상
  * 고전적 시계열 특성
    * 간단한 시계열 특성 : 평균, 확산(표준편차), 이상점(Outliers), 분포
    * 고급 시계열 특성
      * 자기상관 (Autocorrelation) : 자신과 시차가 이쓴 버전의 통계적 상관관계 측정
      * 푸리에 분석 : 시계열을 데이터 집합에서 발생하는 주파수 범위 내 Sin, Cos 함수의 합계로 분해
      * 이산 푸리에 변환 : 시계열의 스펙트럼 밀도를 주파수 함수로 계산
      * 주기도 : 주기를 나타낸 그림
    * 시계열 모델 예
      * 자기회귀
      * 자기회귀 이동 평균
      * 가치 모델
      * 은닉 마르코프 모델


# 8장 고급 자연 언어 처리 예제: 영화 감상평 평점


# 9장 머신러닝 작업 흐름 확장


# 10장 예제: 디지털 디스플레이 광고


# 참고자료
* [https://github.com/brinkar/real-world-machine-learning](https://github.com/brinkar/real-world-machine-learning)
* [https://www.manning.com/books/real-world-machine-learning](https://www.manning.com/books/real-world-machine-learning)
* [http://wikibook.co.kr](http://wikibook.co.kr)
