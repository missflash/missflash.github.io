---
title: "Bayesian Methods for Hackers"
date: 2019. 4. 2. 오후 10:45:33
categories:
use_math: true
---

![Bayesian Methods for Hackers](https://raw.githubusercontent.com/missflash/missflash.github.io/master/_files/bayesian_methods_for_hackers.jpg)


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




# 3. MCMC 블랙박스 열기


# 4. 아무도 알려주지 않는 위대한 이론


# 5. 오히려 큰 손해를 보시겠습니까?


# 6. 우선순위 바로잡기


# 7. 베이지안 A/B 테스트


# 부록 A


# 참고자료
* [https://github.com/CamDavidsonPilon/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers](https://github.com/CamDavidsonPilon/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers)
* [https://github.com/gilbutITbook/006775](https://github.com/gilbutITbook/006775)
* [https://nbviewer.jupyter.org/github/CamDavidsonPilon/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers/tree/master/](https://nbviewer.jupyter.org/github/CamDavidsonPilon/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers/tree/master/)
