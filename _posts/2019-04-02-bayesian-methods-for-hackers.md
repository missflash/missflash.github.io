---
title: "Bayesian Methods for Hackers"
date: 2019. 4. 2. 오후 10:45:33
categories:
use_math: true
toc: true
toc_label: "Table of Contents"
toc_icon: "cog"
toc_sticky: true
---

![Bayesian Methods for Hackers](https://raw.githubusercontent.com/missflash/missflash.github.io/master/_files/bayesian_methods_for_hackers.jpg){: width="30%" height="30%"}


# 1. 베이지안 추론의 철학
* 서론
  * 사전확률 : 사건 A에 대한 우리의 믿음의 양<br>
$P(A)$<br>
  * 사후확률 : 증거 X가 주어진 상황에서 A의 확률<br>
$P(A|X)$<br>
* 베이지안 프레임워크
  * 베이즈 정리<br>
$P(A|X) = \frac {p(A)p(X|A)}{p(X)}$<br>

* 확률분포
  * 확률변수 Z가 이산적인 경우, 확률질량함수<br>
$P(Z=k) = \frac {\lambda^k e^{-\lambda}}{k!}$, $k=0,1,2,...$<br>
$Z$ ~ $Poi(\lambda)$<br>
$E[Z|\lambda] = \lambda$<br>

  * 확률변수 Z가 연속적인 경우, 확률밀도함수<br>
$f_Z(z|\lambda) = \lambda e^{-\lambda z}, z \ge 0$<br>
$Z$ ~ $Exp(\lambda)$<br>
$E[Z|\lambda] = \frac {1}{\lambda}$<br>

* 컴퓨터를 사용하여 베이지안 추론하기
  * [PyMC를 사용한 베이지안](https://nbviewer.jupyter.org/github/CamDavidsonPilon/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers/blob/master/Chapter1_Introduction/Ch1_Introduction_PyMC3.ipynb)
* 결론
* 부록
* 연습문제
* 참고자료


# 2. PyMC 더 알아보기
* 서론
  * Stochastic 변수 : 값이 정해지지 않은 변수
  * Deterministic 변수 : 변수의 부모를 모두 알고있는 경우에 랜덤하지 않은 변수
    * @pymc.deterministic 파이썬 래퍼 (데코레이터)를 써서 구분
* 모델링 방법
  * 관측된 빈도와 실제 빈도간에는 차이가 발생
  * 베르누이 분포<br>
$X$ ~ $Berp(p)$ : X는 p의 확률로 1, 1-p의 확률로 0<br>
  * 이항분포<br>
$X$ ~ $Bin(N, p)$<br>
$P(X=k)=\binom {n} {k} p^k (1-p)^{N-k}$<br>
기댓값 : $Np$<br>
  * 데이터 Import 예시<br>
<script src="https://gist.github.com/missflash/f7dc4640fb695217997a3766c6ef0223.js"></script>
  * Logistic Function<br>
<script src="https://gist.github.com/missflash/c3f69cb3ced7ca2d178bec16fa42a4ce.js"></script>
![Logistic Function](https://raw.githubusercontent.com/missflash/missflash.github.io/master/_files/logistic_function.png)
  * 정규분포<br>
정규확률변수 : $X \sim N(\mu, 1/\tau)$<br>
확률밀도함수 : $f(x|\mu,\tau)=\sqrt{\frac {\tau}{2\pi}}exp(-\frac{\tau}{2}(x-\mu)^2)$<br>
정규분포의 기댓값 : $E[X|\mu, \tau]=\mu$<br>
정규분포의 분산 : $Var(X|\mu, \tau)=\frac{1}{\tau}$<br>

* 우리의 모델이 적절한가?
* 결론
* 부록
* 연습문제
* 참고자료


# 3. MCMC 블랙박스 열기
* 베이지안 지형
* 수렴 판정하기<br>
<script src="https://gist.github.com/missflash/aa7d16e3f87f17664366f3998b2be1fe.js"></script>
* MCMC에 대한 유용한 팁
* 결론
* 참고자료


# 4. 아무도 알려주지 않는 위대한 이론
* 서론
* 큰 수의 법칙
  * 직관<br>
$\frac{1}{N} \sum_{i=1}^N \;Z_i
& =\frac{1}{N} \big(  \sum_{ Z_i = c_1}c_1 + \sum_{Z_i=c_2}c_2 \big) \\\\[5pt]
& = c_1 \sum_{ Z_i = c_1}\frac{1}{N} + c_2 \sum_{ Z_i = c_2}\frac{1}{N} \\\\[5pt]
& = c_1 \times \text{ (approximate frequency of $c_1$) } \\\\ 
& \;\;\;\;\;\;\;\;\; + c_2 \times \text{ (approximate frequency of $c_2$) } \\\\[5pt]
& \approx c_1 \times P(Z = c_1) + c_2 \times P(Z = c_2 ) \\\\[5pt]
& = E[Z]$<br>
<script src="https://gist.github.com/missflash/a26e2ca33fd371a54a05eb6c4ada3edc.js"></script>
* 작은 수의 혼란
* 결론
* 부록
* 연습문제
* 참고자료


# 5. 오히려 큰 손해를 보시겠습니까?
* 서론
* 손실함수
* 베이지안 방법을 통한 기계학습
* 결론
* 참고자료


# 6. 우선순위 바로잡기
* 서론
* 주관적인 사전확률분포 vs. 객관적인 사전확률분포
* 알아두면 유용한 사전확률분포
* 예제: 베이지안 MAB (Multi-Armed-Bandits)<br>
<script src="https://gist.github.com/missflash/e935b305a4c405f1fd0cee79b7212d89.js"></script>
* 해당 분야 전문가로부터 사전확률분포 유도하기
* 켤레 사전확률분포
* 제프리 사전확률분포
* N이 증가할 때 사전확률분포의 효과
* 결론
* 부록
* 참고자료


# 7. 베이지안 A/B 테스트
* 서론
* 전환율 테스트 개요
* 선형손실함수 추가하기
* 전환율을 넘어서: t-검정
* 증분 추정하기
* 결론
* 참고자료


# 부록 A
* 파이썬, PyMC
* 주피터 노트북
* Reddit 실습하기


# 참고자료
* [https://github.com/CamDavidsonPilon/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers](https://github.com/CamDavidsonPilon/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers)
* [https://github.com/gilbutITbook/006775](https://github.com/gilbutITbook/006775)
* [https://nbviewer.jupyter.org/github/CamDavidsonPilon/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers/tree/master/](https://nbviewer.jupyter.org/github/CamDavidsonPilon/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers/tree/master/)
