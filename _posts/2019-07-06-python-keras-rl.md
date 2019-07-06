---
title: "RL with Python and Keras"
date: 2019. 7. 6. 오전 11:04:01
categories:
use_math: true
---

![RL with Python and Keras](https://raw.githubusercontent.com/missflash/missflash.github.io/master/_files/python_keras_rl.jpg)


# 1. 강화학습 개요
* 순차적 행동 결정 문제의 구성요소
  * 상태 (State)
  * 행동 (Action)
  * 보상 (Reward)
  * 정책 (Policy)
* MDP (Markov Decision Process)

# 2. 강화학습 기초 1: MDP와 벨만 방정식
* MDP의 구성요소
  * 상태<br>
$S_t = s$ : 시간 t에서의 상태<br>
  * 행동<br>
$A_t = a$ : 시간 t에서의 행동<br>
  * 보상 함수<br>
$R^a_s = E[R_{t+1} | S_t=s,A_t=a]$ : 시간 t에서 상태가 s이고 행동이 a일때 에이전트가 받을 보상<br>
  * 상태 변환 확률 (State Transition Probability)<br>
$P^a_{ss} = P[S_{t+1}=s | S_t=s,A_t=a]$ : 상태 s에서 행동 a를 취했을 때 다른 상태 s에 도달할 확률<br>
  * 감가율 (Discount Factor)<br>
$\gamma\in[0,1]$<br>
$\gamma^{k-1}R_{t+k}$ : 현재 시간 t로부터 k가 지난후의 보상<br>
  * 정책<br>
$\pi(a|s)=P[A_t=a|S_t=s]$ : 시간 t, 상태 s의 에이전트가 있을 때 가능한 행동 중에서 행동 a를 할 확률<br>

* 가치함수<br>
$R_{t+1}+R_{t+2}+R_{t+3}+R_{t+4}+R_{t+5}+\cdots$ : 시간 t로부터 에이전트가 행동을 하면서 받을 보상들의 합<br>
$G_t=R_{t+1}+\gamma R_{t+2}+\gamma^2 R_{t+3}+\gamma^3 R_{t+4}+\gamma^4 R_{t+5}+\cdots$ : 시간 t로부터 에이전트가 행동을 하면서 받을 보상들의 반환값<br><br>
ex) 에피소드를 t=1 ~ 5까지 진행했을 경우, 아래와 같은 5개의 반환값이 생김<br>
$G_1=R_2+\gamma R_3+\gamma^2 R_4+\gamma^3 R_5+\gamma^4 R_6$<br>
$G_2=R_3+\gamma R_4+\gamma^2 R_5+\gamma^3 R_6$<br>
$G_3=R_4+\gamma R_5+\gamma^2 R_6$<br>
$G_4=R_5+\gamma R_6$<br>
$G_5=R_6$<br><br>
$v(s)=E[G_t|S_t=s]$ : 반환값의 기대값으로 표현된 가치함수<br>
$v(s)=E[R_{t+1}+\gamma R_{t+2}+\gamma^2 R_{t+3}+\gamma^3 R_{t+4}+\gamma^4 R_{t+5}+\cdots|S_t=s]$<br>
$v(s)=E[R_{t+1}+\gamma (R_{t+2}+\gamma R_{t+3}+\gamma^2 R_{t+4}+\gamma^3 R_{t+5}+\cdots)|S_t=s]$<br>
$v(s)=E[R_{t+1}+\gamma G_{t+1}|S_t=s]$<br>
$v(s)=E[R_{t+1}+\gamma v(S_{t+1})|S_t=s]$ : $G_{t+1}$ 은 앞으로 받을 보상에 대한 기대값이기 때문에 가치함수로 표현 가능<br>
$v_\pi(s)=E_\pi[R_{t+1}+\gamma v_\pi(S_{t+1})|S_t=s]$ : 정책을 고려한 가치함수의 표현 (벨만 기대 방정식)<br>

* 큐함수
  * 행동 가치함수 (큐함수) : 어떤 상태에서 어떤 행동이 얼마나 좋은지 알려주는 함수<br>
$v_\pi(s)=\Sigma_{a\in A}\pi(a|s)q_\pi(s,a)$<br>
$q_\pi(s,a)=E_\pi[R_{t+1}+\gamma q_\pi(S_{t+1},A_{t+1})|S_t=s,A_t=a]$<br>

* 벨만 방정식
  * 벨만 기대 방정식<br>
$v_\pi(s)=E_\pi[R_{t+1}+\gamma v_\pi(S_{t+1})|S_t=s]$ : 벨만 기대 방정식<br>
$v_\pi(s)=\Sigma_{a\in A}\pi(a|s)(R_{t+1}+\gamma \Sigma_{s'\in S}P^a_{ss'}v_\pi(s'))$ : 벨만 기대 방정식<br>
$v_\pi(s)=\Sigma_{a\in A}\pi(a|s)(R_{t+1}+\gamma v_\pi(s'))$ : 상태 변환 확률이 1인 벨만 기대 방정식<br>
  * 벨만 최적 방정식<br>
$v_{k+1}(s)=\Sigma_{a\in A}\pi(a|s)(R^a_s+\gamma v_k(s'))$ : 계산 가능한 형태의 벨만 기대 방정식<br>
$v_*(s)=\max_a[q_*(s,a)|S_t=s,A_t=a]$ : 큐함수 중 최대를 선택하는 최적 가치함수<br>
$v_*(s)=\max_a E[R_{t+1}+\gamma v_*(S_{t+1})|S_t=s,A_t=a]$ : 벨만 최적 방정식<br>
$q_*(s,a)=E[R_{t+1}+\gamma max_{a'} q_*(S_{t+1},a')|S_t=s,A_t=a]$ : 큐함수에 대한 벨만 최적 방정식<br>

# 3. 강화학습 기초 2: 그리드월드와 다이내믹 프로그래밍
*

# 4. 강화학습 기초 3: 그리드월드와 큐러닝
*

# 5. 강화학습 심화 1: 그리드월드와 근사함수
*

# 6. 강화학습 심화 2: 카트폴
*

# 7. 강화학습 심화 3: 아타리
*

# 참고자료
* [https://github.com/rlcode/reinforcement-learning-kr](https://github.com/rlcode/reinforcement-learning-kr)
