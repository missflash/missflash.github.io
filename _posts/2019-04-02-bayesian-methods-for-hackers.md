---
title: "Bayesian Methods for Hackers"
date: 2019. 4. 2. 오후 10:45:33
categories:
use_math: true
---

![Think Bayes](https://raw.githubusercontent.com/missflash/missflash.github.io/master/_files/bayesian_methods_for_hackers.jpg)


# 1. 베이지안 추론의 철학
* 서론
  * 사전확률 $P(A)$ : 사건 A에 대한 우리의 믿음의 양
  * 사후확률 $P(A|X)$ : 증거 X가 주어진 상황에서 A의 확률
* 베이지안 프레임워크
  * 베이즈 정리

$P(A|X) = \frac {p(A)p(X|A)}{p(X)}$

* 확률분포
  * 확률변수 Z가 이산적인 경우, 확률질량함수

$P(Z=k) = \frac {\lambda^k e^{-k}}{k!}$, $k=0,1,2,...$<br>
$Z$ ~ $Poi(\lambda)$<br>
$E[Z|\lambda] = \lambda$

  * 확률변수 Z가 연속적인 경우, 확률밀도함수

$f_Z(z|\lambda) = \lambda e^{-\lambda z}, z \ge 0$<br>
$Z$ ~ $Exp(\lambda)$<br>
$E[Z|\lambda] = \frac {1}{\lambda}$

* 컴퓨터를 사용하여 베이지안 추론하기
  * [PyMC를 사용한 베이지안](https://nbviewer.jupyter.org/github/CamDavidsonPilon/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers/blob/master/Chapter1_Introduction/Ch1_Introduction_PyMC3.ipynb)


# 2. PyMC 더 알아보기


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
