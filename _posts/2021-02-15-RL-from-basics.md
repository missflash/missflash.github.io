---
title: "RL from Basics"
date: 2021. 2. 15. 오전 11:04:01
categories:
use_math: true
toc: true
toc_label: "Table of Contents"
toc_icon: "cog"
toc_sticky: true
---

[comment]: <> (포스트 화면 넓게 설정하고 싶을 때 추가, classes: wide)

![RL from Basics (바닥부터 배우는 강화학습)](https://raw.githubusercontent.com/missflash/missflash.github.io/master/_files/RL_from_basics.jpg){: width="30%" height="30%"}


# 1. 강화학습이란
* 1.1 지도학습과 강화학습
* 1.2 순차적 의사결정 문제
* 1.3 보상
  * 보상
  * 누적 보상
  * 스칼라 vs. 벡터
  * 희소하고 지연된 보상
  * 밸류 네트워크
* 1.4 에이전트와 환경
  * 에이전트
  * 환경
  * 상태
  * 상태 변화
  * 연속적 vs. 이산적
* 1.5 강화학습의 위력
  * 병렬성의 힘
  * 자가학습 (Self-Learning)의 매력



# 2. 마르코프 결정 프로세스 (Markov Decision Process)
* 2.1 마르코프 프로세스 (Markov Process)
  * 마르코프 프로세스 정의
    * 미리 정의된 확률 분포를 따라서 상태와 상태사이를 이동해 다니는 여정<br>
$MP\equiv(S,P)$<br>
  * 마르코프 프로세스 구성요소
    * 상태의 집합 $S$<br>
    * 전이 확률 행렬 $P_{SS'}$<br>
  * 마르코프 성질<br>
$P[S_{t+1}|S_t]=P[S_{t+1}|S_1,S_2,\cdots,S_t]$
* 2.2 마르코프 리워드 프로세스 (Markov Reward Process)
  * 마르코프 리워드 프로세스 정의
    * 마르코프 프로세스에 보상의 개념이 추가<br>
$MRP\equiv(S,P,R,\gamma)$<br>
  * 마르코프 리워드 프로세스 구성요소
    * 상태의 집합 $S$<br>
    * 전이 확률 행렬 $P_{SS'}$<br>
    * 보상함수<br>
$R=E[R_t|S_t=s]$<br>
    * 감쇠인자 $\gamma$<br>
  * 에피소드
  * 리턴<br>
$G_t=R_{t+1}+\gamma R_{t+2}+\gamma^2 R_{t+3}+\cdots$<br>
  * 감쇠인자는 왜 필요할까?
    * 수학적 편의성
    * 사람의 선호 반영 (즉각적인 보상 선호)
    * 미래에 대한 불확실성 반영
  * 밸류와 기댓값
  * 에피소드 샘플링 (Monte-Carlo 접근법)
  * 상태 가치 함수<br>
$v(s)=E[G_t|S_t=s]$<br>
* 2.3 마르코프 결정 프로세스 (Markov Decision Process)
  * 마르코프 결정 프로세스 정의
    * 마르코프 리워드 프로세스에 에이전트 개념이 추가<br>
$MDP\equiv(S,A,P,R,\gamma)$<br>
  * 마르코프 결정 프로세스 구성요소
    * 상태의 집합 $S$<br>
    * 액션의 집합 $A$<br>
    * 전이 확률 행렬<br>
$P^a_{SS'}=P[S_{t+1}=s'|S_t=s,A_t=a]$<br>
    * 보상함수<br>
$R^a_s=E[R_{t+1}|S_t=s,A_t=a]$<br>
    * 감쇠인자 $\gamma$<br>
  * 정책함수와 2가지 가치함수
    * 정책함수<br>
$\pi(a|s)=P[A_t=a|S_t=s]$<br>
    * 상태 가치함수<br>
$v_\pi(s)=E_\pi[r_{t+1}+\gamma r_{t+2}+\gamma^2 r_{t+3}+\cdots|S_t=s]=E_\pi[G_t|S_t=s]$<br>
    * 액션 가치함수<br>
$q_\pi(s,a)=E_\pi[G_t|S_t=s,A_t=a]$<br>
* 2.4 Prediction과 Control
  * Prediction : 정책 $\pi$가 주어졌을 때 각 상태의 밸류를 평가하는 문제
  * Control : 최적 정책 $\pi^*$ 를 찾는 문제
  * 최적 가치함수



# 3. 벨만 방정식
* 3.1 벨만 기대 방정식
  * 재귀함수
  * 0단계<br>
$v_\pi(s_t)=E_\pi[r_{t+1}+\gamma v_\pi(s_{t+1})]$<br>
$q_\pi(s_t,a_t)=E_\pi[r_{t+1}+\gamma q_\pi(s_{t+1},a_{t+1})]$<br>
  * 1단계<br>
$v_\pi(s)=\sum_{a\in A}\pi(a|s)q_\pi(s,a)$<br>
$q_\pi(s,a)=r^a_s+\gamma \sum_{s'\in S}P^a_{ss'}v_\pi(s')$<br>
  * 2단계<br>
$v_\pi(s)=\sum_{a\in A}\pi(a|s)\left(r^a_s+\gamma \sum_{s'\in S}P^a_{ss'}v_\pi(s')\right)$<br>
$q_\pi(s,a)=r^a_s+\gamma \sum_{s'\in S}P^a_{ss'}\sum_{a'\in A}\pi(a'|s')q_\pi(s',a')$<br>

* 3.2 벨만 최적 방정식
  * 최적 밸류와 최적 정책<br>
$v_* (s)=\max_{\pi} v_\pi(s)$<br>
$q_* (s,a)=\max_{\pi} q_\pi(s,a)$<br>
  * 0단계<br>
$v_* (s_t)=\max_a E[r_{t+1}+\gamma v_ * (s_{t+1})]$<br>
$q_* (s_t,a_t)=E[r_{t+1}+\gamma \max_{a'} q_ * (s_{t+1},a')]$<br>
  * 1단계<br>
$v_* (s)=\max_a q_ * (s,a)$<br>
$q_* (s,a)=r^a_s+\gamma \sum_{s' \in S}P^a_{ss'}v_ * (s')$<br>
  * 2단계<br>
$v_* (s)=\max_a \left[r^a_s+\gamma \sum_{s' \in S}P^a_{ss'}v_ * (s')\right]$<br>
$q_* (s,a)=r^a_s+\gamma \sum_{s' \in S}P^a_{ss'}\max_a'q_ * (s',a')$<br>



# 4. MDP를 알 때의 플래닝
* 4.1 밸류 평가하기 - 반복적 정책 평가
  * 플래닝 : MDP에 대한 모든 정보를 알 때, 이를 이용하여 정책을 개선해나가는 과정
  * 테이블 기반 방법론
* 4.2 최고의 정책 찾기 - 정책 이터레이션
  * 벨만 기대 방정식 1단계 활용  
  * 그리디 정책 (Greedy Policy)
  * 정책 평가<br>
$V\rightarrow V_\pi$<br>
  * 정책 개선<br>
$\pi \rightarrow \pi_{greedy}$<br>
  * Early Stopping
* 4.3 최고의 정책 찾기 - 밸류 이터레이션
  * 벨만 최적 방정식 1단계 활용  



# 5. MDP를 모를 때 밸류 평가하기
* 5.1 몬테카를로 학습
  * 모델 프리
  * 몬테카를로 방법론 예시 : MCTS (Monte Carlo Tree Search), MCMC (Markov Chain Monte Carlo)
  * 대수의 법칙 (Law of large numbers)
  * 알고리즘 : 테이블 초기화, 경험 쌓기, 테이블 업데이트, 밸류 계산<br>
$v_\pi(s_t)\cong\frac{V(s_t)}{N(s_t)}$<br>
$v(s_t)\leftarrow v(s_t)+\alpha (G_t-V(s_t))$<br>
  * [몬테카를로 코드](https://github.com/seungeunrho/RLfrombasics/blob/master/ch5_MCLearning.py)
* 5.2 Temporal Difference 학습
  * 추측을 추측으로 업데이트하자
  * 이론적 배경<br>
$v_\pi(s_t)=E_\pi[G_t]$<br>
$G_t$ 는 $v_\pi(s_t)$ 의 불편 추정량<br>
$v_\pi(s_t)=E_\pi[r_{t+1}+\gamma v_\pi(s_{t+1})]$<br>
TD Target : $r_{t+1}+\gamma v_\pi(s_{t+1})$<br>
  * 알고리즘<br>
MC : $v(s_t)\leftarrow v(s_t)+\alpha (G_t-V(s_t))$<br>
TD : $v(s_t)\leftarrow v(s_t)+\alpha (r_{t+1}+\gamma \boldsymbol{V(s_{t+1})}-V(s_t))$<br>
  * [TD 코드](https://github.com/seungeunrho/RLfrombasics/blob/master/ch5_TDLearning.py)
* 5.3 몬테카를로 vs TD
  * 학습 시점
    * Episodic MDP : MC, TD 사용 가능
    * Non-Episodic MDP : TD만 사용 가능
  * 편향성 (Bias)
    * MC : 불편 추정량
    * 실제 TD Target (불편 추정량)과 우리가 사용하는 TD Target (편향)은 다른 값<br>
$V(s_{t+1})\ne v_\pi(s_{t+1})$<br>
  * 분산 (Variance)
    * MC : 수십 ~ 수백 개의 확률적 결과, 분산이 큼
    * TD : 한 Step만 예측, 분산이 작음
* 5.4 몬테카를로와 TD의 중간?
  * n Step TD<br>
$N=n:r_{t+1}+\gamma r_{t+2}+\gamma^2 r_{t+3}+\cdots+\gamma^nV(s_{t+n})$<br>



# 6. MDP를 모를 때 최고의 정책 찾기
* 6.1 몬테카를로 컨트롤
  * MDP를 모르기 때문에 보상과 전이 확률 행렬을 알 수 없음
$v_\pi(s_t)=\sum_{a\in A}\pi(a|s)\left(r^a_s+\gamma \sum_{s'\in S}P^a_{ss'}v_\pi(s')\right)$<br>
  * 해결방법
    * 평가 자리에 MC 사용
    * V 대신 Q 사용
    * Greedy 대신 $\epsilon$-Greedy 사용 (Decaying $\epsilon$-Greedy)
  * [모델 프리 MC](https://github.com/seungeunrho/RLfrombasics/blob/master/ch6_MCControl.py)
* 6.2 TD 컨트롤 1 - SARSA
  * MC vs. TD<br>
TD로 V 학습 : $V(S)\leftarrow V(S) + \alpha (R+\gamma V(S')-V(S))$<br>
TD로 Q 학습 : $Q(S,A)\leftarrow Q(S,A) + \alpha (R+\gamma Q(S',A')-Q(S,A))$<br>
  * [모델 프리 SARSA](https://github.com/seungeunrho/RLfrombasics/blob/master/ch6_SARSA.py)
* 6.3 TD 컨트롤 2 - Q러닝
  * On-Policy : Target Policy와 Behavior Policy가 같은 경우
  * Off-Policy : Target Policy와 Behavior Policy가 다른 경우
  * Off-Policy의 장점
    * 과거의 경험 재사용 가능
    * 사람의 데이터로부터 학습 가능
    * 일대다, 다대일 학습 가능
  * 이론적 배경<br>
$q_* (s,a)=\max_{\pi}q_\pi(s,a)$<br>
$\pi_* =argmax_a q_ * (s,a)$<br>
$q_* (s,a)=r^a_s+\gamma \sum_{s' \in S}P^a_{ss'}\max_a'q_ * (s',a')$<br>
$q_* (s,a)=E[r+\gamma \max_{a'}q_ * (s,a')]$<br>
SARSA : $Q(S,A)\leftarrow Q(S,A) + \alpha (R+\gamma Q(S',A')-Q(S,A))$<br>
Q Learning : $Q(S,A)\leftarrow Q(S,A) + \alpha (R+\gamma \max_{A'}Q(S',A')-Q(S,A))$<br>
  * SARSA vs. Q Learning
    * Behavior Policy : SARSA, Q Learning 모두 $\epsilon$-Greedy 사용
    * Target Policy : SARSA는 $\epsilon$-Greedy, Q Learning은 Greedy 사용<br>
SARSA : $q_\pi(s_t,a_t)=E_\pi[r_{t+1}+\gamma q_\pi(s_{t+1},a_{t+1})]$<br>
Q Learning : $q_* (s,a)=E_{s'}[r+\gamma \max_{a'}q_ * (s',a')]$<br>
    * $E_\pi$는 정책함수 $\pi$를 따라가는 경로에 대해 기댓값 계산
    * $E_{s'}$는 정책함수 $\pi$와 전혀 관련없음 (어떠한 정책을 사용해도 무관)
  * [모델 프리 Q Learning](https://github.com/seungeunrho/RLfrombasics/blob/master/ch6_QLearning.py)



# 7. Deep RL 첫 걸음
* 7.1 함수를 활용한 근사
  * 이산적 상태 vs. 연속적인 상태 공간
  * 근사 함수<br>
$f(s)=v_\pi (s)$<br>
  * 최소제곱법
  * 평균제곱오차 (Mean Squared Error)
  * 다항함수<br>
$f(x)=a_0+a_1x+a_2x^2+\cdots+a_nx^n$<br>
  * 오버피팅 vs. 언더피팅
  * 함수의 장점 : 일반화
* 7.2 인공 신경망의 도입
  * 인공신경망의 구성요소
    * Input/Output Layer
    * Hidden Layer
    * Node
  * Linear Combination
  * Non-Linear Activation
  * RELU (Rectified Linear Unit)
  * Loss Function
  * Partial Derivative
  * Gradient<br>
$\nabla_wL(w)=(\frac{\partial L(w)}{\partial w_1},\frac{\partial L(w)}{\partial w_2},\cdots,\frac{\partial L(w)}{\partial w_n})$<br>
  * Gradient Descent<br>
$w'=w-\alpha*\nabla_wL(w)$<br>
  * 미니 배치
  * [신경망 구현 사례](https://github.com/seungeunrho/RLfrombasics/blob/master/ch7_CosineFitting.py)



# 8. 가치 기반 에이전트
* 8.1 밸류 네트워크의 학습
  * 가치 기반 에이전트
  * 정책 기반 에이전트
  * Actor-Critic
  * 밸류 네트워크<br>
$L(\theta)=E_\pi[(v_{true}(s)-v_\theta (s))^2]$<br>
$\nabla_\theta L(\theta)=-E_\pi[(v_{true}(s)-v_\theta (s))\nabla_\theta v_\theta(s)]$<br>
$\nabla_\theta L(\theta)\approx-(v_{true}(s)-v_\theta (s))\nabla_\theta v_\theta(s)$<br>
$\theta'=\theta-\alpha\nabla_\theta L(\theta)$<br>
$\theta'=\theta+\alpha(\boldsymbol{v_{true}(s)}-v_\theta (s))\nabla_\theta v_\theta(s)$<br>
  * 몬테카를로 리턴<br>
$\theta'=\theta+\alpha(\boldsymbol{G_t}-v_\theta (s_t))\nabla_\theta v_\theta(s_t)$<br>
  * TD Target<br>
$\theta'=\theta+\alpha(\boldsymbol{r_{t+1}+\gamma v_\theta(s_{t+1})}-v_\theta (s_t))\nabla_\theta v_\theta(s_t)$<br>
* 8.2 딥 Q러닝
  * 가치 기반 에이전트
    * 명시적 정책이 따로 없음
    * 내재된 정책 사용 (액션-가치 함수 Q)
  * 이론적 배경<br>
$Q_* (s,a)=E_{s'}[r+\gamma max_{a'}Q_ * (s',a')]$<br>
$Q(s,a)\leftarrow Q(s,a)+\alpha(\boldsymbol{r+\gamma max_{a'}Q(s',a')}-Q(s,a))$<br>
$L(\theta)=E[(\boldsymbol{r+\gamma max_{a'}Q_\theta(s',a')}-Q_\theta(s,a))^2]$<br>
$\theta'=\theta+\alpha(\boldsymbol{r+\gamma max_{a'}Q_\theta(s',a')}-Q_\theta(s,a))\nabla_\theta Q_\theta(s,a)$<br>
  * 미니 배치
  * 딥 Q러닝 Pseudo Code
    * 1) $Q_\theta$ 의 파라미티 $\theta$ 초기화<br>
    * 2) 에이전트의 상태 $s$ 를 초기화 ($s\leftarrow s_0$)<br>
    * 3) 에피소드가 끝날때까지 A~E 반복<br>
      * A) $Q_\theta$ 에 대한 $\epsilon-greedy$ 를 이용해 액션 $a$ 선택<br>
      * B) $a$ 를 실행하여 $r$ 과 $s'$ 관측<br>
      * C) $s'$ 에서 $Q_\theta$ 에 대한 $Greedy$ 를 이용해 액션 $a'$ 선택<br>
      * D) $\theta$ 업데이트 : $\theta\leftarrow\theta+\alpha(r+\gamma Q_\theta(s',a')-Q_\theta(s,a))\nabla_\theta Q_\theta(s,a)$<br>
      * E) $s\leftarrow s'$<br>
    * 에피소드가 끝나면 다시 2로 돌아가서 $\theta$ 가 수렴할 때까지 반복<br>
  * Experience Replay
    * 상태 전이 : $e_t=(s_t,a_t,r_t,s_{t+1})$<br>
    * 리플레이 버퍼 : 낱개의 데이터 재사용 (선입선출)
    * 상관성 억제 : 다양한 데이터 섞임 (Shuffle)
  * Target Network<br>
$L(\theta)=E[(R+\gamma max_{A'}Q_{\theta_{i}^{-}}(S',A')-Q_{\theta_i} (S,A))^2]$<br>
$Q_{\theta_{i}^{-}}$ : Target Network<br>
$Q_{\theta_i}$ : Q Network<br>
일정 주기마다 $\theta_{i}^{-} \leftarrow \theta_i$<br>
    * 뉴럴 네트워크를 학습할 때 정답지가 자주 변하는 것은 학습의 안정성을 떨어뜨리기 때문
  * [DQN 구현 사례](https://github.com/seungeunrho/RLfrombasics/blob/master/ch8_DQN.py)



# 9. 정책 기반 에이전트
* 9.1 Policy Gradient
  * 결정론적 정책 vs. 확률론적 정책
  * Gradient Ascent<br>
$J(\theta)=E_{\pi_\theta}[\sum_tr_t]=v_{\pi_\theta}(s_0)$<br>
$J(\theta)=\sum_{s\in S}d(s)*v_{\pi_\theta}(s)$<br>
$\theta'\leftarrow\theta+\alpha\nabla_\theta J(\theta)$<br>
  * 1-Step MDP<br>
  $J(\theta)=\sum_{s\in S}d(s)*v_{\pi_\theta}(s)$<br>
  $J(\theta)=\sum_{s\in S}d(s)\sum_{a\in A}\pi_\theta(s,a) *R_{s,a}$<br>
  $\nabla_\theta J(\theta)=\nabla_\theta \sum_{s\in S}d(s)\sum_{a\in A}\pi_\theta(s,a) *R_{s,a}$<br>
  $\nabla_\theta J(\theta)=\sum_{s\in S}d(s)\sum_{a\in A}\nabla_\theta\pi_\theta(s,a) *R_{s,a}$<br>
  $\nabla_\theta J(\theta)=\sum_{s\in S}d(s)\sum_{a\in A}\frac{\pi_\theta(s,a)}{\pi_\theta(s,a)}\nabla_\theta\pi_\theta(s,a) *R_{s,a}$<br>
  $\nabla_\theta J(\theta)=\sum_{s\in S}d(s)\sum_{a\in A}\pi_\theta(s,a)\nabla_\theta log \pi_\theta(s,a) *R_{s,a}$<br>
  $\nabla_\theta J(\theta)=E_{\pi_\theta}[\nabla_\theta log \pi_\theta(s,a) *R_{s,a}]$<br>
  * MDP<br>
$\nabla_\theta J(\theta)=E_{\pi_\theta}[\nabla_\theta log \pi_\theta(s,a)*Q_{\pi_\theta}(s,a)]$<br>
* 9.2 REINFORCE 알고리즘
  * 이론적 배경<br>
$\nabla_\theta J(\theta)=E_{\pi_\theta}[\nabla_\theta log \pi_\theta(s,a)*G_t]$<br>
$Q_{\pi_\theta}(s,a)=E[G_t|s_t=s,a_t=a]$<br>
  * REINFORCE Pseudo Code<br>
    * 1) $\pi_\theta(s,a)$ 의 파라미터 $\theta$ 를 랜덤으로 초기화<br>
    * 2) 에피소드가 끝날때까지 A~C 반복<br>
      * A) 에이전트의 상태를 초기화 : $s\leftarrow s_0$<br>
      * B) $\pi_\theta$ 를 이용하여 에피소드 끝까지 진행 ${s_0,a_0,r_0,s_1,a_1,r_1,\cdots,s_T,a_T,r_T}$<br>
      * C) $t=0~T$ 에 대해 다음 반복<br>
        * $G_t\leftarrow \sum^T_{i=t} r_i*\gamma^{i-t}$<br>
        * $\theta\leftarrow \theta+\alpha\nabla_\theta log \pi_\theta(s_t,a_t)*G_t$ : $s$ 에서 $a$ 를 선택할 확률을 $G_t$ 에 비례하게 증가<br>
  * 참고<br>
$\nabla_\theta J(\theta)\neq E_{\pi_\theta}[\nabla_\theta \pi_\theta(s,a) *G_t]$<br>
$\nabla_\theta J(\theta)\approx G_t *\nabla_\theta log \pi_\theta(s_t,a_t)$<br>
$L(\theta)=(\boldsymbol{r+\gamma max_{a'}Q_\theta(s',a')}-Q_\theta(s,a))^2$<br>
$\nabla_\theta L(\theta)\approx -(\boldsymbol{r+\gamma max_{a'}Q_\theta(s',a')}-Q_\theta(s,a))\nabla_\theta Q_\theta(s,a)$<br>
$J(\theta)=G_t *log \pi_\theta(s_t,a_t)$ : Maximize 하고 싶은 값 (Gradient Ascent 사용)<br>
$J(\theta)=-G_t *log \pi_\theta(s_t,a_t)$ : Minimize 하고 싶은 값 (Gradient Descent 사용)<br>
* 9.3 액터-크리틱
  * Q 액터-크리틱<br>
$\nabla_\theta J(\theta)=E_{\pi_\theta}[\nabla_\theta log \pi_\theta(s,a)*Q_{\pi_\theta}(s,a)]$<br>
  * 액터 : 실행할 액션 $a$ 를 선택하는 $\pi_\theta$<br>
  * 크리틱 : 실행할 액션 $a$ 의 밸류를 평가하는 $Q_w$<br>
  * Q Actor-Critic Pseudo Code<br>
    * 1) 정책, 액션-밸류 네트워크의 파라미터 $\theta$ 와 $w$ 초기화<br>
    * 2) 상태 $s$ 초기화<br>
    * 3) 액션 $a \sim \pi_\theta(a|s)$<br>
    * 4) 에피소드가 끝날때까지 A~E 반복<br>
      * A) $a$ 를 실행하여 보상 $r$ 과 다음 상태 $s'$ 을 얻음<br>
      * B) $\theta$ 업데이트 : $\theta\leftarrow \theta+\alpha\nabla_\theta log \pi_\theta(s,a)*Q_w(s,a)$<br>
      * C) 액션 $a' \sim \pi_\theta(a'|s')$<br>
      * D) $w$ 업데이트 : $w\leftarrow w+\beta(r+\gamma Q_w(s',a')-Q_w(s,a))\nabla_w Q_w(s,a)$<br>
      * E) $a\leftarrow a', s\leftarrow s'$<br>
  * 어드밴티지 액터-크리틱<br>
$A_{\pi_\theta}(s,a)\equiv Q_{\pi_\theta}(s,a)-V_{\pi_\theta}(s)$<br>
$A_{\pi_\theta}(s,a)$ : Advantage<br>
$V_{\pi_\theta}(s)$ : Baseline (기저)<br>
$E_{\pi_\theta}[\nabla_\theta log \pi_\theta(s,a) *B(s)]=\sum_{s\in S}d_{\pi_\theta}(s)\sum_{a\in A}\pi_\theta(s,a)\nabla_\theta log \pi_\theta(s,a) *B(s)$<br>
$\sum_{s\in S}d_{\pi_\theta}(s)\sum_{a\in A}\pi_\theta(s,a)\nabla_\theta log \pi_\theta(s,a) *B(s)=\sum_{s\in S}d_{\pi_\theta}(s)\sum_{a\in A}\pi_\theta(s,a)\frac{\nabla_\theta \pi_\theta(s,a)}{\pi_\theta(s,a)}*B(s)$<br>
$\sum_{s\in S}d_{\pi_\theta}(s)\sum_{a\in A}\pi_\theta(s,a)\nabla_\theta log \pi_\theta(s,a) *B(s)=\sum_{s\in S}d_{\pi_\theta}(s)\sum_{a\in A}\nabla_\theta \pi_\theta(s,a) *B(s)$<br>
$\sum_{s\in S}d_{\pi_\theta}(s)\sum_{a\in A}\pi_\theta(s,a)\nabla_\theta log \pi_\theta(s,a) *B(s)=\sum_{s\in S}d_{\pi_\theta}(s)B(s)\sum_{a\in A}\nabla_\theta \pi_\theta(s,a)$<br>
$\sum_{s\in S}d_{\pi_\theta}(s)\sum_{a\in A}\pi_\theta(s,a)\nabla_\theta log \pi_\theta(s,a) *B(s)=\sum_{s\in S}d_{\pi_\theta}(s)B(s)\nabla_\theta\sum_{a\in A} \pi_\theta(s,a)$<br>
$\sum_{s\in S}d_{\pi_\theta}(s)\sum_{a\in A}\pi_\theta(s,a)\nabla_\theta log \pi_\theta(s,a) *B(s)=\sum_{s\in S}d_{\pi_\theta}(s)B(s)\nabla_\theta1=0$<br>
$\therefore E_{\pi_\theta}[\nabla_\theta log \pi_\theta(s,a) *B(s)]=0$<br>
$\nabla_\theta J(\theta)=E_{\pi_\theta}[\nabla_\theta log \pi_\theta(s,a)*A_{\pi_\theta}(s,a)]$<br>
$A_{\pi_\theta}(s,a)=Q_{\pi_\theta}(s,a)-V_{\pi_\theta}(s)$<br>
$Q_{\pi_\theta}(s,a)\approx Q_w$<br>
$V_{\pi_\theta}(s)\approx V_\phi(s)$<br>
    * 정책함수 $\pi_\theta(s,a)$ 의 뉴럴넷 $\theta$<br>
    * 액션-가치함수 $Q_w(s,a)$ 의 뉴럴넷 $w$<br>
    * 가치함수 $V_\phi$ 의 뉴럴넷 $\phi$<br>
  * 어드밴티지 액터-크리틱 Pseudo Code<br>
    * 1) 3쌍의 뉴럴넷 파라미터 $\theta,w,\phi$ 초기화<br>
    * 2) 상태 $s$ 초기화<br>
    * 3) 액션 $a \sim \pi_\theta(a|s)$ 를 샘플링<br>
    * 4) 에피소드가 끝날때까지 A~F 반복<br>
      * A) $a$ 를 실행하여 보상 $r$ 과 다음 상태 $s'$ 을 얻음<br>
      * B) $\theta$ 업데이트 : $\theta\leftarrow \theta+\alpha_1\nabla_\theta log \pi_\theta(s,a)*\{Q_w(s,a)-V_\phi(s)\}$<br>
      * C) 액션 $a' \sim \pi_\theta(a'|s')$ 를 샘플링<br>
      * D) $w$ 업데이트 : $w\leftarrow w+\alpha_2(r+\gamma Q_w(s',a')-Q_w(s,a))\nabla_w Q_w(s,a)$<br>
      * E) $\phi$ 업데이트 : $\phi\leftarrow \phi+\alpha_3(r+\gamma V_\phi(s')-V_\phi(s))\nabla_\phi V_\phi(s)$<br>
      * F) $a\leftarrow a', s\leftarrow s'$<br>
  * TD 액터-크리틱<br>
$\delta=r+\gamma V(s')-V(s)$<br>
$E_\pi[\delta|s,a]=E_\pi[r+\gamma V(s')-V(s)|s,a]$<br>
$E_\pi[\delta|s,a]=E_\pi[r+\gamma V(s')|s,a]-V(s)$<br>
$E_\pi[\delta|s,a]=Q(s,a)-V(s)=A(s,a)$ : $\delta$ 는 $A(s,a)$ 의 불편추정량<br>
$\nabla_\theta J(\theta)=E_{\pi_\theta}[\nabla_\theta log \pi_\theta(s,a)*\delta]$<br>
  * TD Actor-Critic Pseudo Code<br>
    * 1) 정책, 밸류 네트워크 파라미터 $\theta,\phi$ 초기화<br>
    * 2) 액션 $a \sim \pi_\theta(a|s)$ 를 샘플링<br>
    * 3) 에피소드가 끝날때까지 A~E 반복<br>
      * A) $a$ 를 실행하여 보상 $r$ 과 다음 상태 $s'$ 을 얻음<br>
      * B) $\delta$ 계산 : $\delta\leftarrow r+\gamma V_\phi(s')-V_\phi(s)$<br>
      * C) $\theta$ 업데이트 : $\theta\leftarrow \theta+\alpha_1\nabla_\theta log \pi_\theta(s,a)*\delta$<br>
      * D) $\phi$ 업데이트 : $\phi\leftarrow \phi+\alpha_2\delta\nabla_\phi V_\phi(s)$<br>
      * E) $a\leftarrow a', s\leftarrow s'$<br>
  * [TD Actor-Critic 구현 사례](https://github.com/seungeunrho/RLfrombasics/blob/master/ch9_ActorCritic.py)
    * loss function 계산시, detach() 함수 반영사실에 주의
```
loss = -torch.log(pi_a) * delta.detach() + F.smooth_l1_loss(self.v(s), td_target.detach())
```
    * delta.detach()는 $\delta$ 를 상수처리 해서 업데이트 되지 않도록 하기 위함<br>
    * td_target.detach()는 TD Target을 상수처리 해서 업데이트 되지 않도록 하기 위함<br>



# 10. 알파고와 MCTS
* 10.1 알파고
* 10.2 알파고 제로



# 11. 블레이드 & 소울 비무 AI 만들기
* 11.1 블레이드 & 소울 비무
* 11.2 비무에 강화학습 적용하기
* 11.3 전투 스타일 유도를 통한 새로운 방식의 Self-Play 학습



# 참고자료
* [https://github.com/seungeunrho/RLfrombasics](https://github.com/seungeunrho/RLfrombasics)
