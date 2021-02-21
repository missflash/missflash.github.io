---
title: "Real World Machine Learning"
date: 2019. 3. 25. 오후 8:56:55
categories:
use_math: true
toc: true
toc_label: "Table of Contents"
toc_icon: "cog"
toc_sticky: true
---

![리얼월드 머신러닝](https://raw.githubusercontent.com/missflash/missflash.github.io/master/_files/real_world_machine_learning.jpg){: width="30%" height="30%"}


# 1장 머신러닝이란 무엇인가?
* 머신러닝의 다섯 가지 이점
  * 정확성
  * 자동화
  * 신속성
  * 맞춤화
  * 확장성
* 모델 성능 최적화
  * 모델 매개변수 조정 (Tuning the model parameters)
  * 특성의 부차 집합을 선택 (Selecting a subset of features)
  * 데이터 전처리 (Preprocessing the data)



# 2장 실무현장 데이터
* [Github Notebook](https://nbviewer.jupyter.org/github/brinkar/real-world-machine-learning/blob/master/Chapter%202%20-%20Data%20Processing.ipynb)
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
* [Github Notebook](https://nbviewer.jupyter.org/github/brinkar/real-world-machine-learning/blob/master/Chapter%203%20-%20Modeling%20and%20prediction.ipynb)
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
* [Github Notebook](https://nbviewer.jupyter.org/github/brinkar/real-world-machine-learning/blob/master/Chapter%204%20-%20Evaluation%20and%20Optimization.ipynb)
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
* [Github Notebook](https://nbviewer.jupyter.org/github/brinkar/real-world-machine-learning/blob/master/Chapter%206%20-%20NYC%20Taxi%20Full%20Example.ipynb)
* 데이터
  * 데이터 시각화
  * 머신러닝 작업에서의 두 가지 함정
    * 믿기지 않는 시나리오 (Too-Good-To-Be-True Scenario)
    * 섣부른 가정 (Premature Assumptions)
* 모델링
  * 선형 모델
  * 비선형 분류기 (Ex. 랜덤 포레스트 등)



# 7장 고급 특성 추출 기법
* [Github Notebook](https://nbviewer.jupyter.org/github/brinkar/real-world-machine-learning/blob/master/Chapter%207%20-%20Advanced%20Feature%20Engineering.ipynb)
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
* [Github Notebook](https://nbviewer.jupyter.org/github/brinkar/real-world-machine-learning/blob/master/Chapter%208%20-%20Movie%20Review%20Full%20Example.ipynb)
* 데이터 및 사용사례 탐구
  * 사용사례의 목적
    * 목표 변수를 변환하는 방법 (Ex. 2진 분류, 다중 분류, 실제값 등)
    * 최적화할 평가 기준
    * 고려할 학습 알고리즘
    * 데이터 입력 사용 가능 여부
  * 사용사례 고려사항
    * 사용사례가 가치 있는 이유는?
    * 어떤 훈련 데이터가 필요한가?
    * 적절한 머신러닝 모델링 전략은?
    * 예측에 어떤 평가 측정을 사용해야 하는가?
    * 가지고 있는 데이터가 이 사용사례를 해결하기에 충분한가?
* 기초 자연 언어 처리 특성 추출 및 초기 모델 구축
  * 나이브 베이즈 알고리즘<br>
$p(C_k|x)$ ~ $p(C_k)p(x|C_k)$<br>
$p(C_k|x)$ ~ $p(C_k)p(x_1|C_k)p(x_2|C_k)p(x_3|C_k)...$<br>
$p(C_k|x)$ ~ $p(C_k)\prod_i^n p(x_i|C_k)$<br>
$p(x_i|C_k)$ ~ $\prod_i$ $p_{k_i}^{x_i}$<br>
$log[p(C_k|x_i)]$ ~ $log[p(C_k)\prod_i$ $p_{k_i}^{x_i}]$<br>
$log[p(C_k|x_i)]$ ~ $log[p(C_k)]+\sum_i^n x_i log(p_{k_i})$<br>
$log[p(C_k|x_i)]$ ~ $b+w_kx$<br>
    * $b$는 데이터를 통해 알수 있고, $x$는 예측하고자 하는 사례의 특성을 나타냄, $w_k$는 좋은 문서 또는 나쁜 문서에서 단어가 출연하는 비율을 의미
    * 초기 모델 평가하기
<script src="https://gist.github.com/missflash/80fd6c8fe74a7f9ef9b7b594c6584704.js"></script>
  * tf-idf 알고리즘으로 단어 주머니의 특성들을 정규화하기
    * Scikit-learn의 TfidfVectorizer 활용
  * 모델 매개변수 최적화
* 고급 알고리즘과 모델 배치 고려사항
  * word2vec 특성
    * Gensim 라이브러리의 word2vec 활용
  * 랜덤 포레스트 모델
    * RandomForestClassifier 활용



# 9장 머신러닝 작업 흐름 확장
* [Github Notebook](https://nbviewer.jupyter.org/github/brinkar/real-world-machine-learning/blob/master/Chapter%209%20-%20Scaling%20ML%20Workflows.ipynb)
* 확장하기 전에
  * 중요 차원 식별 : 자원 제약이 모델 훈련과 예측에 미치는 영향
  * 훈련 데이터 Subsampling 하기
    * 특성 선택 : N-그램, Lasso 등
    * 사례 군집화 : 계층적 병합 군집화 등
  * 확장 가능한 데이터 관리 시스템
    * 수평 확장 : 새로운 노트 추가 및 노드간 고르게 데이터 분포
    * 수직 확장 : 메모리, CPU 코어 추가 등
    * Hadoop Distributed File System (HDFS) : MapReduce 알고리즘 사용
    * 데이터 집약성 (Data Locality)
    * 머아웃 (Mahout) : HDFS와 함께 작동하는 머신러닝 알고리즘
    * 아파치 스파크 : 하둡 기반 AI 플랫폼
    * MLlib : 머신러닝 알고리즘 및 도구
* 머신러닝 모델링 파이프라인 확장
  * 학습 알고리즘 확장
    * 다항식 특성 : 특성1*특성2, 특성1의 제곱 등
    * 데이터와 알고리즘 근사 : 히스토그램 근사
    * 심층 신경망 : 딥러닝과 같은 블랙박스 모델
* 예측 확장
  * 예측량 높이기
  * 예측속도 높이기



# 10장 예제: 디지털 디스플레이 광고
* [Github Notebook](https://nbviewer.jupyter.org/github/brinkar/real-world-machine-learning/blob/master/Chapter%2010%20-%20Advertising%20Example.ipynb)
* 특성 추출과 모델링 전략
  * 고차원 공간으로 인한 차원의 저주 (The Curse of Dimensionality)
* 데이터의 크기와 모양
  * Cardinality : 집합의 크기 또는 사상 개수
* 특잇값 분해
  * Singular Value Decomposition (SVD)<br>
$A_{n~by~p}=U_{n~by~n}S_{n~by~p}V_{p~by~p}^T$<br>
$U$ 왼쪽 특이 벡터, $V$ 오른쪽 특이 벡터<br>
$S$ 특잇값 : 해당 특성 벡터가 어느정도 독립적인지 확인 가능<br>
* 자원 추정 및 최적화
  * 병렬 처리를 통한 데이터 수집 시간 단축
* k 최근접 이웃
  * 훈련 공간에서 가장 가까운 관측값에 대한 예측이 목적
  * 유클리드 거리 사용
* 랜덤 포레스트
  * 다중 결정 트리 분류기나 회귀기를 훈련 데이터와 특성 부분집합에 맞추고 결합된 모델을 토대로 예측할 수 있는 앙상블 학습방법
  * Bootstrap Aggregating
  * Bagging (집어넣기) : 랜덤 포레스트와 다른 알고리즘에 사용되는 복원 표집 반복 과정
  * Stacking (쌓기) : 최종 합의된 예측을 도출하기 위해 로지스틱 회귀 같은 다른 알고리즘 예측을 결합하는 방법



# 부록. 인기 있는 머신러닝 알고리즘
* Linear Regression (선형 회귀)
* Logistic Regression (로지스틱 회귀)
* Support Vector Machine (서포트 벡터 머신)
* SVM with Kernel (커널을 가진 서포트 벡터 머신)
* k-nearest neighbors (k 최근접 이웃)
* Decision Trees (결정 트리)
* Random Forest (랜덤 포레스트)
* Boosting (부스팅)
* Naive Bayes (나이브 베이즈)
* Neural Network (신경망)
* Vowpal Wabbit (보우팰 웨빗)
* XGBoost (엑스지부스트)



# 참고자료
* [https://github.com/brinkar/real-world-machine-learning](https://github.com/brinkar/real-world-machine-learning)
* [https://www.manning.com/books/real-world-machine-learning](https://www.manning.com/books/real-world-machine-learning)
* [http://wikibook.co.kr](http://wikibook.co.kr)
