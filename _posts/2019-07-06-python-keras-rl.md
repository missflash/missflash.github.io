---
title: "RL with Python and Keras"
date: 2019. 7. 6. 오전 11:04:01
categories:
use_math: true
toc: true
toc_label: "Table of Contents"
toc_icon: "cog"
---

[comment]: <> (포스트 화면 넓게 설정하고 싶을 때 추가, classes: wide)

![RL with Python and Keras](https://raw.githubusercontent.com/missflash/missflash.github.io/master/_files/python_keras_rl.jpg){: width="30%" height="30%"}


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
$P^a_{ss'} = P[S_{t+1}=s' | S_t=s,A_t=a]$ : 상태 s에서 행동 a를 취했을 때 다른 상태 s'에 도달할 확률<br>
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

* 벨만 기대 방정식<br>
$v_\pi(s)=E_\pi[R_{t+1}+\gamma v_\pi(S_{t+1})|S_t=s]$ : 벨만 기대 방정식<br>
$v_\pi(s)=\Sigma_{a\in A}\pi(a|s)(R_{t+1}+\gamma \Sigma_{s'\in S}P^a_{ss'}v_\pi(s'))$ : 계산 가능한 벨만 기대 방정식<br>
$v_\pi(s)=\Sigma_{a\in A}\pi(a|s)(R_{t+1}+\gamma v_\pi(s'))$ : 상태 변환 확률이 1인 벨만 기대 방정식<br>

* 벨만 최적 방정식<br>
$v_{k+1}(s)=\Sigma_{a\in A}\pi(a|s)(R^a_s+\gamma v_k(s'))$ : 계산 가능한 형태의 벨만 기대 방정식<br>
$v_\ast(s)=\max_\pi[v_\pi(s)]$ : 최적의 가치함수<br>
$q_\ast(s,a)=\max_\pi[q_\pi(s,a)]$ : 최적의 큐함수<br>
$\pi_\ast(s,a)=\cases{1\space \text{if}\space a=argmax_{a\in A} q_\ast(s,a)\cr 0\space \text{otherwise}}$ : 최적 정책<br>
$v_\ast(s)=\max_a[q_\ast(s,a)|S_t=s,A_t=a]$ : 큐함수 중 최대를 선택하는 최적 가치함수<br>
$v_\ast(s)=\max_a E[R_{t+1}+\gamma v_\ast(S_{t+1})|S_t=s,A_t=a]$ : 벨만 최적 방정식<br>
$q_\ast(s,a)=E[R_{t+1}+\gamma \max_{a'} q_\ast(S_{t+1},a')|S_t=s,A_t=a]$ : 큐함수에 대한 벨만 최적 방정식<br>

# 3. 강화학습 기초 2: 그리드월드와 다이내믹 프로그래밍
* 정책 이터레이션<br>
$v_\pi(s)=E_\pi[R_{t+1}+\gamma v_\pi(S_{t+1})|S_t=s]$ : 벨만 기대 방정식을 통한 효율적인 가치함수 계산<br>
$v_\pi(s)=\Sigma_{a\in A}\pi(a|s)(R_{t+1}+\gamma v_\pi(s'))$ : 합의 형태로 표현한 벨만 기대 방정식<br>
$v_{k+1}(s)=\Sigma_{a\in A}\pi(a|s)(R_s^a+\gamma v_k(s'))$ : k번째 가치함수를 통해 k+1번째 가치함수 계산<br>
  * [environment](https://github.com/missflash/reinforcement-learning-kr/blob/master/1-grid-world/1-policy-iteration/environment.py)
  * [policy_iteration](https://github.com/missflash/reinforcement-learning-kr/blob/master/1-grid-world/1-policy-iteration/policy_iteration.py)
* 다이나믹 프로그래밍의 한계
  * 다이나믹 프로그래밍 : 순차적 행동 결정 문제를 벨만 방정식을 통해 푸는 것
  * 계산 복잡도
  * 차원의 저주
  * 환경에 대한 완벽한 정보 필요


# 4. 강화학습 기초 3: 그리드월드와 큐러닝
* 강화학습과 정책 평가 1: 몬테카를로 예측<br>
$v_\pi(s)\thicksim \frac {1}{N(s)}\Sigma_{i=1}^{N(s)}G_i(s)$ : 반환값($G$)의 평균으로 가치함수($v$)를 추정<br>
$V_{n+1}=\frac {1}{n}\Sigma_{i=1}^{n}G_i=\frac {1}{n}(G_n+\Sigma_{i=1}^{n-1}G_i)$<br>
$=\frac {1}{n}(G_n+(n-1)\frac {1}{n-1}\Sigma_{i=1}^{n-1}G_i)$<br>
$=\frac {1}{n}(G_n+(n-1)V_n)$<br>
$=V_n+\frac {1}{n}(G_n-V_n)$<br>
$V(s)\gets V(s)+\frac {1}{n}(G(s)-V(s))$<br>
$V(s)\gets V(s)+\alpha(G(s)-V(s))$<br>
* 강화학습과 정책 평가 2: 시간차 예측<br>
$v_\pi(s)=E_\pi[R_{t+1}+\gamma v_\pi(S_{t+1})|S_t=s]$ : 정책을 고려한 가치함수의 표현 (벨만 기대 방정식)<br>
$V(S_t)\gets V(S_t)+\alpha(R+\gamma V(S_{t+1})-V(S_t))$ : 시간차 예측에서 가치함수 업데이트<br>
$R+\gamma V(S_{t+1})$ : 업데이트의 목표<br>
$\alpha(R+\gamma V(S_{t+1})-V(S_t))$ : 업데이트의 크기<br>
* 강화학습과 알고리즘 1: 살사
  * 정책 이터레이션 (GPI : Generalized Policy Iteration)<br>

$\pi'(s)=argmax_{a \in A} [R_s^a+\gamma P^a_{ss'}V(s')]$ : GPI의 탐욕 정책 발전<br>
$\pi(s)=argmax_{a \in A} Q(s,a)$ : 큐함수를 사용한 탐욕 정책<br>
$Q(S_t,A_t)\gets Q(S_t,A_t)+\alpha(R+\gamma Q(S_{t+1},A_{t+1})-Q(S_t,A_t))$ : 시간차 예측에서 큐함수 업데이트<br>
$[S_t,A_t,R_{t+1},S_{t+1},A_{t+1}]$ : 시간차 제어에서 사용하는 샘플<br>
  * 탐욕정책 : 초기의 에이전트는 탐욕 정책으로 잘못된 학습을 하게 될 가능성이 큼
$\pi(s)=\cases{a^\ast=argmax_{a\in A} Q(s,a), 1-\varepsilon \cr a \ne a^\ast, \varepsilon}$ : $\varepsilon$-탐욕 정책<br>
  * [살사 코드](https://github.com/rlcode/reinforcement-learning-kr/tree/master/1-grid-world/4-sarsa)
  * 에이전트 작동 방식
    * 현재 상태에서 $\varepsilon$-탐욕 정책에 따라 행동 선택
    * 선택한 행동으로 환경에서 한 타임스텝 진행
    * 환경으로부터 보상과 다음 상태 받음
    * 다음 상태에서 $\varepsilon$-탐욕 정책에 따라 다음 행동 선택
    * $(s,a,r,s',a')$을 통해 큐함수 업데이트
* 강화학습과 알고리즘 2: 큐러닝
  * 살사
    * `온폴리시` 시간차 제어 (On-Policy Temporal-Difference Control) 때문에 자신이 행동하는대로 학습
    * 탐험을 위해 선택한 $\varepsilon$-탐욕 정책때문에 에이전트가 최적 정책을 학습하지 못하는 문제 발생
  * 큐러닝
    * `오프폴리시` 시간차 제어 (Off-Policy Temporal-Difference Control), 또는 큐러닝으로 해결 가능<br>

$Q(S_t,A_t)\gets Q(S_t,A_t)+\alpha(R_{t+1}+\gamma max_{a'}Q(S_{t+1},a')-Q(S_t,A_t))$ : 큐러닝을 통한 큐함수 업데이트<br>
$q^\ast(s,a)=E[R_{t+1}+\gamma max_{a'}q^\ast(S_{t+1},a')|S_t=s,A_t=a]$ : 큐함수에 대한 벨만 최적 방정식<br>
  * [큐러닝 코드](https://github.com/rlcode/reinforcement-learning-kr/tree/master/1-grid-world/4-sarsa)<br>

비슷한 수식들이 여러개 있어 확실한 이해&개념정리 필요!
{: .notice--info}



# 5. 강화학습 심화 1: 그리드월드와 근사함수
* 근사함수
  * 몬테카를로, 살사, 큐러닝의 한계
    * 계산 복잡도
    * 차원의 저주
    * 환경에 대한 완벽한 정보 필요
  * 근사함수를 통한 가치함수의 매개변수화
* 인공신경망
  * 노드와 활성함수
    * 노드의 종류
      * 입력층 (Input Layer)
      * 은닉층 (Hidden Layer)
      * 출력층 (Output Layer)
    * 활성함수의 종류
      * Sigmoid 함수<br>
$\frac{1}{1+e^{-x}}$<br>
      * ReLU 함수<br>
$f(x)=\cases{x, x \geq 0 \cr 0, x < 0}$<br>
  * 딥러닝
    * 특징 추출 (Feature Extraction)
  * 신경망의 학습<br>
$오차 = (타깃 - 예측)^2$<br>
$업데이트 값 \varpropto 오차 * 오차 기여도$<br>
    * 경사하강법의 종류
      * SGD
      * RMSprop
      * Adam
* 케라스
* 딥살사
  * 살사의 큐함수 업데이트 식<br>
$Q(S_t,A_t)\gets Q(S_t,A_t)+\alpha(R_{t+1}+\gamma Q(S_{t+1},A_{t+1})-Q(S_t,A_t))$<br>
  * 살사의 큐함수 업데이트 식에서 정답<br>
$R_{t+1}+\gamma Q(S_{t+1},A_{t+1})$<br>
  * 살사의 큐함수 업데이트 식에서 예측<br>
$Q(S_t,A_t)$<br>
  * 딥살사의 오차함수<br>
$MSE = (정답 - 예측)^2 = (R_{t+1}+\gamma Q(S_{t+1},A_{t+1}) - Q(S_t,A_t))^2$<br>
  * 딥살사의 활성함수 : 선형함수
  * [딥살사 코드](https://github.com/rlcode/reinforcement-learning-kr/tree/master/1-grid-world/6-deep-sarsa)<br>
* 정책신경망 (Policy Gradient)
  * 정책신경망의 활성함수 : Softmax<br>
$s(y_i)=\frac{e^{y_i}} {\Sigma_j e^{y_j}}$<br>
  * 정책의 표현<br>
$정책 = \pi_\theta (a|s) : \theta = 가중치$<br>
  * 정책 기반 강화학습의 목표<br>
$maximize J(\theta)$<br>
  * 목표함수의 경사에 따라 정책신경망 업데이트<br>
$\theta_{t+1} = \theta_t + \alpha \nabla_\theta J(\theta)$<br>
  * 가치함수로 표현한 목표함수의 정의<br>
$J(\theta) = v_{\pi_\theta} (s_0)$<br>
  * 목표함수의 미분<br>
$\nabla_\theta J(\theta) = \nabla_\theta v_{\pi_\theta} (s_0)$<br>
  * Policy Gradient<br>
$\nabla_\theta J(\theta) = \sum_s d_{\pi_\theta}(s) \sum_a \nabla_\theta \pi_\theta (a|s) q_\pi (s,a)$<br>
$\nabla_\theta J(\theta) = \sum_s d_{\pi_\theta}(s) \sum_a \pi_\theta(a|s) \frac {\nabla_\theta \pi_\theta (a|s)} {\pi_\theta (a|s)} q_\pi (s,a)$<br>
$\nabla_\theta J(\theta) = \sum_s d_{\pi_\theta}(s) \sum_a \pi_\theta(a|s) \nabla_\theta log \pi_\theta (a|s) q_\pi (s,a)$<br>
  * 기댓값의 형태로 표현한 목표함수의 미분<br>
$\nabla_\theta J(\theta) = E_{\pi_\theta} [\nabla_\theta log \pi_\theta (a|s) q_\pi (s,a)]$<br>
  * Policy Gradient의 업데이트 식<br>
$\theta_{t+1} = \theta_t + \alpha \nabla_\theta J(\theta) \approx \theta_t + \alpha[\nabla_\theta log \pi_\theta (a|s) q_\pi (s,a)]$<br>
  * 강화학습의 업데이트 식<br>
$\theta_{t+1} \approx \theta_t + \alpha[\nabla_\theta log \pi_\theta (a|s) G_t]$<br>
  * [강화학습 코드](https://github.com/rlcode/reinforcement-learning-kr/tree/master/1-grid-world/7-reinforce)<br>
  * 에이전트가 환경과 상호작용하는 방법
    * 상태에 따른 행동 선택
    * 선택한 행동으로 환경에서 한 타임스텝 진행
    * 환경으로부터 다음 상태와 보상 받음
    * 다음 상태에 대한 행동 선택 (에피소드가 끝날 때까지 반복)
    * 환경으로부터 받은 정보를 토대로 에피소드마다 학습 진행
  * 보상들의 정산 (반환값)<br>
$G_1=R_2+\gamma R_3+\gamma^2 R_4+\gamma^3 R_5+\gamma^4 R_6$<br>
$G_2=R_3+\gamma R_4+\gamma^2 R_5+\gamma^3 R_6$<br>
$G_3=R_4+\gamma R_5+\gamma^2 R_6$<br>
$G_4=R_5+\gamma R_6$<br>
$G_5=R_6$<br>
  * 효율적인 반환값 계산<br>
$G_5=R_6$<br>
$G_4=R_5+\gamma R_5$<br>
$G_3=R_4+\gamma R_4$<br>
$G_2=R_3+\gamma R_3$<br>
$G_1=R_2+\gamma R_2$<br>
  * 네트워크 업데이트를 위한 오류함수<br>
$\nabla_\theta log \pi_\theta (a|s) G_t$<br>
$\nabla_\theta [log \pi_\theta (a|s) G_t]$<br>
  * 엔트로피<br>
$엔트로피 = -\sum_i p_i log p_i$<br>
  * 크로스 엔트로피<br>
$크로스 엔트로피 = -\sum_i y_i log p_i : y_i = 정답$<br>
  * Policy Gradient의 크로스 엔트로피<br>
$크로스 엔트로피 = -\sum_i y_i log p_i = -log~p_{action}$<br>
  * 강화학습의 업데이트 식<br>
$\theta_{t+1} \approx \theta_t + \alpha[\nabla_\theta log \pi_\theta (a|s) G_t] = \theta_t - \alpha[\nabla_\theta -log \pi_\theta (a|s) G_t]$<br>



# 6. 강화학습 심화 2: 카트폴
* 알고리즘1: DQN
  * [Playing Atari with Deep Reinforcement Learning](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf)
  * 경험 리플레이 : 에이전트가 환경에서 탐험하며 얻은 샘플 $(s, a, r, s')$ 을 메모리에 저장
  * 리플레이 메모리 : (과거) 샘플을 저장하는 메모리 → 오프폴리시 알고리즘
  * 타깃 신경망<br>
$Q(S_t,A_t)\gets Q(S_t,A_t)+\alpha(R_{t+1}+\gamma max_{a'}Q(S_{t+1},a')-Q(S_t,A_t))$ : 큐러닝의 큐함수 업데이트 식<br>
$MSE = (정답 - 예측)^2 = (R_{t+1}+\gamma max_{a'}Q(s',a',\theta) - Q(s,a,\theta))^2$ : DQN의 오류함수<br>
$MSE = (정답 - 예측)^2 = (R_{t+1}+\gamma max_{a'}Q(S_{t+1},a',\theta^-) - Q(S_t,A_t,\theta))^2$ : 타깃 네트워크를 이용한 DQN 오류함수<br>
  * [DQN 코드](https://github.com/rlcode/reinforcement-learning-kr/tree/master/2-cartpole/1-dqn)<br>
  * 에이전트가 환경과 상호작용하는 방법
    * 상태에 따른 행동 선택
    * 선택한 행동으로 환경에서 한 타임스텝 진행
    * 환경으로부터 다음 상태와 보상 받음
    * 샘플 $(s, a, r, s')$ 을 리플레이 메모리에 저장
    * 리플레이 메모리에서 무작위 추출한 샘플로 학습
    * 에피소드마다 타깃 모델 업데이트
* 알고리즘2: 액터-크리틱
  * [Reinforcement Learning: Introduction](https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf)<br>
$\theta_{t+1} = \theta_t + \alpha \nabla_\theta J(\theta) \approx \theta_t + \alpha[\nabla_\theta log \pi_\theta (a|s) q_\pi (s,a)]$ : 폴리시 그레디언트의 정책신경망 업데이트 식<br>
$\theta_{t+1} \approx \theta_t + \alpha[\nabla_\theta log \pi_\theta (a|s) G_t]$ : Reinforce 알고리즘의 업데이트 식<br>
$\theta_{t+1} \approx \theta_t + \alpha[\nabla_\theta log \pi_\theta (a|s) Q_w(s,a)]$ : 액터-크리틱 업데이트 식<br>
$오류함수 = 정책\ 신경망\ 출력의\ 크로스\ 엔트로피 * 큐함수(가치신경망\ 출력)$<br>
$A(S_t,A_t)=Q_w(S_t,A_t)-V_v(S_t)$ : 어드밴티지 함수의 정의<br>
$\delta_v=R_{t+1}+\gamma V_v(S_{t+1})-V_v(S_t)$ : 어드밴티지 함수의 정의<br>
$\theta_{t+1} \approx \theta_t + \alpha[\nabla_\theta log \pi_\theta (a|s) \delta_v]$ : 액터-크리틱 업데이트 식<br>
$MSE = (정답 - 예측)^2 = (R_{t+1}+\gamma V_v(S_{t+1})-V_v(S_t))^2$ : 가치신경망의 업데이트를 위한 오류함수<br>
  * A2C (Advantage Actor-Critic)
  * [액터-크리틱 코드](https://github.com/rlcode/reinforcement-learning-kr/tree/master/2-cartpole/2-actor-critic)<br>
  * 에이전트가 환경과 상호작용하는 방법
    * 정책신경망을 통해 확률적으로 행동을 선택
    * 선택한 행동으로 환경에서 한 타임스텝 진행
    * 환경으로부터 다음 상태와 보상 받음
    * 샘플 $(s, a, r, s')$ 을 통해 시간차 에러를 구하고 어드밴티지 함수 구함
    * 시간차 에러로 가치신경망을, 어드밴티지 함수로 정책신경망을 업데이트<br>
$\theta_{t+1} \approx \theta_t + \alpha[\nabla_\theta log \pi_\theta (a|s) \delta_v]$ : 액터 업데이트 식<br>
$\theta_{t+1} \approx \theta_t + \alpha \nabla_\theta [log \pi_\theta (a|s) \delta_v]$ : 액터 업데이트 식 변형<br>
$MSE = (정답 - 예측)^2 = (R_{t+1}+\gamma V_v(S_{t+1})-V_v(S_t))^2$ : 크리틱 오류함수<br>
  * A3C (Asynchronous Advantage Actor-Critic)



# 7. 강화학습 심화 3: 아타리
*

# 참고자료
* [https://github.com/rlcode/reinforcement-learning-kr](https://github.com/rlcode/reinforcement-learning-kr)
