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

![RL from Basics (바닥부터 배우는 강화학습)](https://raw.githubusercontent.com/missflash/missflash.github.io/blob/master/_files/RL_from_basics.jpg){: width="30%" height="30%"}


# 1. 강화학습이란
* 1.1 지도학습과 강화학습
* 1.2 순차적 의사결정 문제
* 1.3 보상
  * 보상
  * 어떻게X, 얼마나O
  * 누적 보상
  * 스칼라 vs. 벡터
  * 희소하고 지연된 보상
  * 밸류 네트워크
* 1.4 에이전트와 환경
  * 에이전트
  * 환경
  * 상태
  * 상태 변화 (환경의 역할)
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
  * 마르코프 성질 : 미래는 오로지 현재에 의해 결정됨<br>
$P[S_{t+1}\mid S_t]=P[S_{t+1}\mid S_1,S_2,\cdots,S_t]$
* 2.2 마르코프 리워드 프로세스 (Markov Reward Process)
  * 마르코프 리워드 프로세스 정의
    * 마르코프 프로세스에 보상의 개념이 추가<br>
$MRP\equiv(S,P,R,\gamma)$<br>
  * 마르코프 리워드 프로세스 구성요소
    * 상태의 집합 $S$<br>
    * 전이 확률 행렬 $P_{SS'}$<br>
    * 보상함수<br>
$R=E[R_t\mid S_t=s]$<br>
    * 감쇠인자 $\gamma$<br>
  * 에피소드
  * 리턴 : 감쇠된 보상의 합<br>
$G_t=R_{t+1}+\gamma R_{t+2}+\gamma^2 R_{t+3}+\cdots$<br>
    * 에피소드
    * 미래에 받을 보상의 합
  * 감쇠인자는 왜 필요할까?
    * 수학적 편의성
    * 사람의 선호 반영 (즉각적인 보상 선호)
    * 미래에 대한 불확실성 반영
  * 밸류와 기댓값
  * 에피소드 샘플링 (Monte-Carlo 접근법)
    * 리턴은 동일 상태에서도 매번 바뀔 수 있음
    * 기댓값 사용 필요
    * 상태 가치 함수
  * 상태 가치 함수<br>
$v(s)=E[G_t\mid S_t=s]$<br>
* 2.3 마르코프 결정 프로세스 (Markov Decision Process)
  * 마르코프 결정 프로세스 정의
    * 마르코프 리워드 프로세스에 에이전트 개념이 추가<br>
$MDP\equiv(S,A,P,R,\gamma)$<br>
  * 마르코프 결정 프로세스 구성요소
    * 상태의 집합 $S$<br>
    * 액션의 집합 $A$<br>
    * 전이 확률 행렬<br>
$P_{SS'}^a=P[S_{t+1}=s'\mid S_t=s,A_t=a]$<br>
    * 보상함수<br>
$R_s^a=E[R_{t+1}\mid S_t=s,A_t=a]$<br>
    * 감쇠인자 $\gamma$<br>
  * 정책함수와 2가지 가치함수
    * 정책함수 : 상태 $s$에서 액션 $a$를 선택할 확률<br>
$\pi(a\mid s)=P[A_t=a\mid S_t=s]$<br>
    * 상태 가치함수<br>
$v_\pi(s)=E_\pi[r_{t+1}+\gamma r_{t+2}+\gamma^2 r_{t+3}+\cdots\mid S_t=s]=E_\pi[G_t\mid S_t=s]$<br>
    * 액션 가치함수<br>
$q_\pi(s,a)=E_\pi[G_t\mid S_t=s,A_t=a]$<br>
* 2.4 Prediction과 Control
  * Prediction : 정책 $\pi$가 주어졌을 때 각 상태의 밸류를 평가하는 문제
  * Control : 최적 정책 $\pi^*$ 를 찾는 문제 (누구를 만나도 다 이기는 정책)
  * 최적 가치함수



# 3. 벨만 방정식
* 3.1 벨만 기대 방정식
  * 모델 프리 : 경험을 통해 학습 (모델을 모르기 때문에 경험을 기반으로 학습해야 함)
  * 모델 기반 : Planning (모델을 알고 있으니까 계획만 수립해도 실제 가치를 평가할 수 있음)
  * 재귀함수
  * 0단계<br>
$v_\pi(s_t)=E_\pi[r_{t+1}+\gamma v_\pi(s_{t+1})]$<br>
$q_\pi(s_t,a_t)=E_\pi[r_{t+1}+\gamma q_\pi(s_{t+1},a_{t+1})]$<br>
  * 1단계<br>
$v_\pi(s)=\sum_{a\in A}\pi(a\mid s)q_\pi(s,a)$<br>
$q_\pi(s,a)=r_s^a+\gamma \sum_{s'\in S}P_{ss'}^av_\pi(s')$<br>
  * 2단계<br>
$v_\pi(s)=\sum_{a\in A}\pi(a\mid s)\left(r_s^a+\gamma \sum_{s'\in S}P_{ss'}^av_\pi(s')\right)$<br>
$q_\pi(s,a)=r_s^a+\gamma \sum_{s'\in S}P_{ss'}^a\sum_{a'\in A}\pi(a'\mid s')q_\pi(s',a')$<br>

* 3.2 벨만 최적 방정식
  * 최적 밸류와 최적 정책<br>
$v_* (s)=\max_{\pi} v_\pi(s)$<br>
$q_* (s,a)=\max_{\pi} q_\pi(s,a)$<br>
  * 0단계<br>
$v_* (s_t)=\max_a E[r_{t+1}+\gamma v_ * (s_{t+1})]$<br>
$q_* (s_t,a_t)=E[r_{t+1}+\gamma \max_{a'} q_ * (s_{t+1},a')]$<br>
  * 1단계<br>
$v_* (s)=\max_a q_ * (s,a)$<br>
$q_* (s,a)=r_s^a+\gamma \sum_{s' \in S}P_{ss'}^av_ * (s')$<br>
  * 2단계<br>
$v_* (s)=\max_a \left[r_s^a+\gamma \sum_{s' \in S}P_{ss'}^av_ * (s')\right]$<br>
$q_* (s,a)=r_s^a+\gamma \sum_{s' \in S}P_{ss'}^a\max_{a'}q_ * (s',a')$<br>



# 4. MDP를 알 때의 플래닝
* 4.1 밸류 평가하기 - 반복적 정책 평가
  * MDP를 안다는 것
    * 보상함수를 알고 있음
    * 전이확률행렬을 알고 있음
    * $S, A, P, R, \gamma$가 주어짐
  * 플래닝 : MDP에 대한 모든 정보를 알 때, 이를 이용하여 정책을 개선해나가는 과정
  * 테이블 기반 방법론
* 4.2 최고의 정책 찾기 - 정책 이터레이션
  * 벨만 기대 방정식 1단계 활용  
  * 그리디 정책 (Greedy Policy)
  * 정책 평가 : 반복적 정책 평가<br>
$V\rightarrow V_\pi$<br>
  * 정책 개선 : 그리디 정책 생성<br>
$\pi \rightarrow \pi_{greedy}$<br>
  * Early Stopping
* 4.3 최고의 정책 찾기 - 밸류 이터레이션
  * MDP를 모두 아는 상황에서는 최적 밸류를 알면 최적 정책을 얻을 수 있음
  * 벨만 최적 방정식 1단계 활용
  * 문제와 MDP 조건에 따른 구분
    * 문제가 작다
      * MDP를 안다
        * 4장
      * MDP를 모른다
        * 5~6장
    * 문제가 크다
      * 7~9장



# 5. MDP를 모를 때 밸류 평가하기
* 5.1 몬테카를로 학습
  * 모델 프리
    * 모델 : 환경의 모델
    * 모델 프리에서의 Prediction
      * 몬테카를로
      * Temporal Difference
    * 가치 함수 : 리턴의 기댓값 (가치 함수의 정의)
  * 몬테카를로 방법론 예시 : MCTS (Monte Carlo Tree Search), MCMC (Markov Chain Monte Carlo)
  * 대수의 법칙 (Law of large numbers)
  * 몬테카를로 학습 알고리즘1 : 테이블 업데이트
    * 테이블 초기화, 경험 쌓기, 테이블 업데이트, 밸류 계산 (에피소드 N개가 모두 끝난 뒤 평균 계산)<br>
$v_\pi(s_t)\cong\frac{V(s_t)}{N(s_t)}$<br>
$V(s_t)\leftarrow V(s_t)+\alpha (G_t-V(s_t))$<br>
  * 몬테카를로 학습 알고리즘2 : 조금씩 업데이트
    * 에피소드가 한 개 끝날때마다 테이블 업데이트
  * [몬테카를로 코드](https://github.com/seungeunrho/RLfrombasics/blob/master/ch5_MCLearning.py)
* 5.2 Temporal Difference 학습
  * TD 학습
    * MC는 반드시 종료하는 MDP에만 사용 가능
    * TD는 추측을 추측으로 업데이트 하는 방식 (Bootstrap)
    * 리턴은 가치 함수의 불편 추정량 (편향되지 않은 추정량)
    * 벨만 기대 방정식의 TD Target 활용
    * 실제 TD Target (불편 추정량)과 우리가 사용한 TD Target (편향)은 같지 않음
  * 이론적 배경<br>
$v_\pi(s_t)=E_\pi[G_t]$<br>
$G_t$ 는 $v_\pi(s_t)$ 의 불편 추정량<br>
$v_\pi(s_t)=E_\pi[r_{t+1}+\gamma v_\pi(s_{t+1})]$<br>
TD Target : $r_{t+1}+\gamma v_\pi(s_{t+1})$<br>
  * 알고리즘<br>
MC : $V(s_t)\leftarrow V(s_t)+\alpha (G_t-V(s_t))$<br>
TD : $V(s_t)\leftarrow V(s_t)+\alpha (r_{t+1}+\gamma \boldsymbol{V(s_{t+1})}-V(s_t))$<br>
  * [TD 코드](https://github.com/seungeunrho/RLfrombasics/blob/master/ch5_TDLearning.py)
* 5.3 몬테카를로 vs TD
  * 학습 시점
    * Episodic MDP : MC, TD 사용 가능
    * Non-Episodic MDP : TD만 사용 가능
  * 편향성 (Bias)
    * MC : 불편 추정량 (가치 함수의 정의로부터 리턴 사용)
    * TD : 벨만 기대 방정식으로부터 TD Target 사용, 실제 TD Target (불편 추정량)과 우리가 사용하는 TD Target (편향)은 다른 값<br>
$V(s_{t+1})\ne v_\pi(s_{t+1})$<br>
  * 분산 (Variance)
    * MC : 수십 ~ 수백 개의 확률적 결과, 분산이 큼
    * TD : 한 Step만 예측, 분산이 작음 (학습에 유리)
* 5.4 몬테카를로와 TD의 중간?
  * n Step TD<br>
$N=n:r_{t+1}+\gamma r_{t+2}+\gamma^2 r_{t+3}+\cdots+\gamma^nV(s_{t+n})$<br>



# 6. MDP를 모를 때 최고의 정책 찾기
* 6.1 몬테카를로 컨트롤
  * 몬테카를로 컨트롤, SARSA, Q Learning
  * 모델 프리에서 정책 이터레이션을 사용할 수 없는 이유 : 1) 보상함수, 전이확률행렬을 모름 (벨만 기대 방정식에서 사용됨), 2) 정책 개선 단계에서 그리디 정책을 만들 수 없음
  * MDP를 모르기 때문에 보상과 전이 확률 행렬을 알 수 없음<br>
$v_\pi(s)=\sum_{a\in A}\pi(a\mid s)\left(r_s^a+\gamma \sum_{s'\in S}P_{ss'}^av_\pi(s')\right)$<br>
  * 해결방법
    * 평가 자리에 MC 사용
    * V 대신 Q 사용
    * Greedy 대신 $\epsilon$-Greedy 사용 (Decaying $\epsilon$-Greedy)
  * 몬테카를로 컨트롤 구현
    * 한 에피소드의 경험 축적
    * 경험한 데이터로 $q(s, a)$ 값 업데이트 (정책 평가)
    * 업데이트 된 $q(s, a)$ 테이블로 $\epsilon$-Greedy 수행 (정책 개선)
  * [모델 프리 MC](https://github.com/seungeunrho/RLfrombasics/blob/master/ch6_MCControl.py)
* 6.2 TD 컨트롤 1 - SARSA
  * MC 대신 TD 사용해서 정책 평가
  * 히스토리 전체가 아니라 샘플 트랜지션 하나만 생기면 업데이트 가능
  * MC vs. TD<br>
TD로 V 학습 : $V(S)\leftarrow V(S) + \alpha (R+\gamma V(S')-V(S))$<br>
TD로 Q 학습 : $Q(S,A)\leftarrow Q(S,A) + \alpha (R+\gamma Q(S',A')-Q(S,A))$<br>
  * [모델 프리 SARSA](https://github.com/seungeunrho/RLfrombasics/blob/master/ch6_SARSA.py)
* 6.3 TD 컨트롤 2 - Q러닝
  * On-Policy : Target Policy와 Behavior Policy가 같은 경우
  * Off-Policy : Target Policy와 Behavior Policy가 다른 경우
  * Target Policy : 강화하고자 하는 목표가 되는 정책
  * Behavior Policy : 환경과 상호작용하며 경험을 쌓고 있는 정책
  * Off-Policy의 장점
    * 과거의 경험 재사용 가능
    * 사람의 데이터로부터 학습 가능
    * 일대다, 다대일 학습 가능
  * 이론적 배경<br>
$q_* (s,a)=\max_{\pi}q_\pi(s,a)$<br>
$\pi_* =argmax_a q_ * (s,a)$<br>
$q_* (s,a)=r_s^a+\gamma \sum_{s' \in S}P_{ss'}^a\max_{a'}q_ * (s',a')$<br>
$q_* (s,a)=E_{s'}[r+\gamma \max_{a'}q_ * (s',a')]$<br>
SARSA : $Q(S,A)\leftarrow Q(S,A) + \alpha (R+\gamma Q(S',A')-Q(S,A))$<br>
Q Learning : $Q(S,A)\leftarrow Q(S,A) + \alpha (R+\gamma \max_{A'}Q(S',A')-Q(S,A))$<br>
  * SARSA vs. Q Learning
    * Behavior Policy : SARSA, Q Learning 모두 $\epsilon$-Greedy 사용
    * Target Policy : SARSA는 $\epsilon$-Greedy, Q Learning은 Greedy 사용<br>
SARSA : $q_\pi(s_t,a_t)=E_\pi[r_{t+1}+\gamma q_\pi(s_{t+1},a_{t+1})]$<br>
Q Learning : $q_* (s,a)=E_{s'}[r+\gamma \max_{a'}q_ * (s',a')]$<br>
    * $E_\pi$는 정책함수 $\pi$를 따라가는 경로에 대해 기댓값 계산
    * $E_{s'}$는 정책함수 $\pi$와 전혀 관련없음 (어떠한 정책을 사용해도 무관)
  * SARSA
    * 벨만 기대 방정식에 기원을 두고 있음
    * 2가지 확률적인 요소 포함
      * $\pi$ : 정책에 의한 확률적인 요소
      * 전이확률행렬 : 환경에 의한 확률적인 요소
  * Q Learning
    * 벨만 최적 방정식에 기원을 두고 있음
    * $\pi$와 관련 없음 : 어떤 정책을 사용해도 상관없음
    * 최적 정책은 환경에 의존적임 : 환경이 정해지면 그에 따라 최적 정책도 결정됨
  * [모델 프리 Q Learning](https://github.com/seungeunrho/RLfrombasics/blob/master/ch6_QLearning.py)



# 7. Deep RL 첫 걸음
* 7.1 함수를 활용한 근사
  * 테이블 방식 접근의 한계
    * 저장 용량의 한계 : 이산적 상태 vs. 연속적인 상태 공간
    * 무한의 상태 방문 불가 : 학습 불가
  * 근사 함수<br>
$f(s)=v_\pi (s)$<br>
  * 최소제곱법
  * 평균제곱오차 (Mean Squared Error)
  * 피팅
  * 다항함수<br>
$f(x)=a_0+a_1x+a_2x^2+\cdots+a_nx^n$<br>
  * 함수의 피팅
    * 함수에 데이터를 기록한다.
    * 데이터 점들을 가장 가깝게 지나도록 함수를 그려본다.
    * 함수의 파라미터 값을 찾는다.
    * 함수를 학습한다.
  * 차수가 높은게 무조건 유리하지 않은 이유
    * 데이터에 노이즈가 섞여 있음
    * 데이터에 확률적 요소가 포함되어 있음
  * 오버 피팅 : 함수가 노이즈에 피팅해버림
  * 언더 피팅 : 함수의 유연성이 부족해 데이터와의 에러가 큼
  * 함수의 장점
    * 일반화
    * 학습의 결과물을 저장하는데 필요한 용량이 적음
* 7.2 인공 신경망의 도입
  * 인공신경망의 구성요소
    * Input/Output Layer
    * Hidden Layer : Node로 구성
  * Node의 동작
    * Linear Combination : Input의 선형 결합을 통해 새로운 Feature 생성 (Input보다 한층 더 추상화된 Feature)
    * Non-Linear Activation : Input과 Output의 비선형 관계 대응 (ReLU, Sigmoid 등)
  * ReLU (Rectified Linear Unit)
  * Loss Function
    * 인공신경망의 학습 : 손실함수의 값이 줄어들도록 파라미터를 수정하는 것
    * $w$가 $L$($w$)에 미치는 영향력 확인 : 미분 필요
    * 편미분 (Partial Derivative)
    * Gradient : 편미분의 벡터 모음
    * 업데이트 크기 : Learning Rate (Step Size)로 결정
  * Gradient<br>
$\nabla_wL(w)=(\frac{\partial L(w)}{\partial w_1},\frac{\partial L(w)}{\partial w_2},\cdots,\frac{\partial L(w)}{\partial w_n})$<br>
  * Gradient Descent<br>
$w'=w-\alpha*\nabla_wL(w)$<br>
  * 학습 절차
    * PyTorch 등에서 자동 미분 라이브러리 활용
    * 역전파 (Backpropagation)을 통해 함수의 Gradient를 빠르게 계산
    * 미니 배치 활용
  * [신경망 구현 사례](https://github.com/seungeunrho/RLfrombasics/blob/master/ch7_CosineFitting.py)



# 8. 가치 기반 에이전트
* 8.1 밸류 네트워크의 학습
  * 가치 기반 에이전트
  * 정책 기반 에이전트
  * Actor-Critic
  * 손실함수의 정의에 기댓값 사용 이유 : 존재하는 모든 상태를 방문할 수 없음, 정책에 의해 자주 방문하는 상태의 가중치가 높아짐 (중요한 상태의 밸류를 더 정확하게 계산할 수 있음)
  * 정책 $\pi$을 이용한 샘플 추출 필요
  * 정답에 해당하는 밸류를 모르기 때문에 몬테카를로 방법의 리턴이나 TD의 TD 타깃 학습 필요
  * 정답에 해당하는 리턴이나 TD 타깃은 상수임에 주의 : 타깃이 변할 경우 학습이 불안정하게 됨, 실제 구현시에는 TD 타깃에 detach 함수 호출하면 됨
  * 밸류 네트워크<br>
$L(\theta)=E_\pi[(v_{true}(s)-v_\theta (s))^2]$<br>
$\nabla_\theta L(\theta)=-E_\pi[(v_{true}(s)-v_\theta (s))\nabla_\theta v_\theta(s)]$<br>
$\nabla_\theta L(\theta)\approx-(v_{true}(s)-v_\theta (s))\nabla_\theta v_\theta(s)$<br>
$\theta'=\theta-\alpha\nabla_\theta L(\theta)$<br>
$\theta'=\theta+\alpha(\boldsymbol{v_{true}(s)}-v_\theta (s))\nabla_\theta v_\theta(s)$ : $v_{true}(s)$ 모름!<br>
  * 몬테카를로 리턴<br>
$\theta'=\theta+\alpha(\boldsymbol{G_t}-v_\theta (s_t))\nabla_\theta v_\theta(s_t)$<br>
  * TD Target<br>
$\theta'=\theta+\alpha(\boldsymbol{r_{t+1}+\gamma v_\theta(s_{t+1})}-v_\theta (s_t))\nabla_\theta v_\theta(s_t)$<br>
* 8.2 딥 Q러닝
  * 가치 기반 에이전트
    * 가치 기반 에이전트에는 명시적 정책이 따로 없음 : 액션 가치 함수 $q(s, a)$ 이용
    * 이런 정책 함수를 내재된 정책이라고 함
  * 이론적 배경<br>
$Q_* (s,a)=E_{s'}[r+\gamma max_{a'}Q_ * (s',a')]$<br>
$Q(s,a)\leftarrow Q(s,a)+\alpha(\boldsymbol{r+\gamma max_{a'}Q(s',a')}-Q(s,a))$<br>
$L(\theta)=E[(\boldsymbol{r+\gamma max_{a'}Q_\theta(s',a')}-Q_\theta(s,a))^2]$<br>
$\theta'=\theta+\alpha(\boldsymbol{r+\gamma max_{a'}Q_\theta(s',a')}-Q_\theta(s,a))\nabla_\theta Q_\theta(s,a)$<br>
  * 미니 배치 : 기댓값 연산자를 없애기 위해 여러개의 샘플을 뽑아서 그 평균을 이용해 업데이트
  * 미니 배치 사이즈
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
  * Off-Policy 학습 : 실행할 액션을 선택하는 행동 정책은 $\epsilon$-Greedy 정책, 학습 대상인 타깃 정책은 Greedy 정책 사용
  * 실제 구현시에는 3-D 대신 $L(\theta)$ 를 사용하면 됨 (라이브러리가 미분 수행)<br>
$L(\theta)=(\boldsymbol{r+\gamma max_{a'}Q_\theta(s',a')}-Q_\theta(s,a))^2$<br>
  * Experience Replay
    * 경험 : 여러개의 에피소드로 구성
    * 에피소드 : 여러개의 상태 전이 (트랜지션)으로 구성
    * 상태 전이 : $e_t=(s_t,a_t,r_t,s_{t+1})$<br>
    * 상관성 억제 : 다양한 데이터 섞임 (Shuffle)
    * 리플레이 버퍼 : 낱개의 데이터 재사용 (선입선출), 가장 최신 N개의 데이터만 유지, 랜덤하게 미니 배치 구성
    * 리플레이 버퍼의 장점 : 여러 데이터가 재사용 될 수 있음 (데이터 효율성 향상), 데이터 사이의 상관성을 낮춤
    * Off-Policy 알고리즘에만 사용 가능
  * Target Network<br>
$L(\theta_i)=E[(R+\gamma max_{A'}Q_{\theta_{i}^{-}}(S',A')-Q_{\theta_i} (S,A))^2]$<br>
$Q_{\theta_{i}^{-}}$ : Target Network<br>
$Q_{\theta_i}$ : Q Network<br>
일정 주기마다 $\theta_{i}^{-} \leftarrow \theta_i$<br>
    * 뉴럴 네트워크를 학습할 때 정답지가 자주 변하는 것은 학습의 안정성을 떨어뜨리기 때문
    * 정답을 계산할 때 사용하는 타깃 네트워크와 학습을 받고 있는 Q 네트워크를 구분해서 사용
    * 타깃 네트워크의 파라미터를 일정 주기마다 업데이트
  * [DQN 구현 사례](https://github.com/seungeunrho/RLfrombasics/blob/master/ch8_DQN.py)



# 9. 정책 기반 에이전트
* 9.1 Policy Gradient
  * 가치 기반 에이전트 : 결정론적 방식으로 액션 결정
  * 정책 기반 에이전트 : 확률론적 방식으로 액션 결정 (좀 더 유연한 정책)
  * 액션 공간이 연속적일 경우 가치 기반 에이전트 사용 불가
  * 환경에 숨겨진 정보가 있거나 환경 자체가 변하는 경우, 정책 기반 에이전트는 유연하게 대처 가능
  * 정책 기반 에이전트에서 정책 함수의 정답을 구하는 것은 어려움, 손실함수 사용 불가, 정책을 평가하는 기준인 평가함수 $J$ 필요
  * 평가함수는 상태의 확률 분포와 가치함수의 곱으로 정의
  * Gradient Ascent<br>
$J(\theta)=E_{\pi_\theta}[\sum_tr_t]=v_{\pi_\theta}(s_0)$<br>
$J(\theta)=\sum_{s\in S}d(s)*v_{\pi_\theta}(s)$<br>
$\theta'\leftarrow\theta+\alpha\nabla_\theta J(\theta)$<br>
  * 모델 프리의 한계
    * 보상함수와 전이확률행렬을 모름
    * 모든 상태에 대해 상태가치함수 합산 불가
    * 기댓값 연산자를 이용해 샘플 기반 방법론으로 풀이 가능!
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
    * 보상함수를 Q로 변경
* 9.2 REINFORCE 알고리즘
  * 이론적 배경<br>
$\nabla_\theta J(\theta)=E_{\pi_\theta}[\nabla_\theta log \pi_\theta(s,a)*G_t]$<br>
$Q_{\pi_\theta}(s,a)=E[G_t\mid s_t=s,a_t=a]$<br>
    * $Q$대신 샘플인 리턴 사용
    * 경험을 통해 리턴이 큰 액션의 확률을 증가시키도록 업데이트
    * Gradient ascent vs. Gradient descent
  * REINFORCE Pseudo Code<br>
    * 1) $\pi_\theta(s,a)$ 의 파라미터 $\theta$ 를 랜덤으로 초기화<br>
    * 2) 에피소드가 끝날때까지 A~C 반복<br>
      * A) 에이전트의 상태를 초기화 : $s\leftarrow s_0$<br>
      * B) $\pi_\theta$ 를 이용하여 에피소드 끝까지 진행 $\{s_0,a_0,r_0,s_1,a_1,r_1,\cdots,s_T,a_T,r_T\}$<br>
      * C) $t=0\sim T$ 에 대해 다음 반복<br>
        * $G_t\leftarrow \sum_{i=t}^T r_i*\gamma^{i-t}$<br>
        * $\theta\leftarrow \theta+\alpha\nabla_\theta log \pi_\theta(s_t,a_t)*G_t$ : $s$ 에서 $a$ 를 선택할 확률을 $G_t$ 에 비례하게 업데이트<br>
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
  * 리턴 대신 Q 사용
  * 액터 : 정책 함수의 가치를 학습하는 방향으로 업데이트, 실행할 액션 $a$ 를 선택하는 $\pi_\theta$<br>
  * 크리틱 : Q 평가 결과가 좋으면 강화, 안좋으면 약화하는 방식으로 업데이트, 실행할 액션 $a$ 의 밸류를 평가하는 $Q_w$<br>
  * Q Actor-Critic Pseudo Code<br>
    * 1) 정책, 액션-밸류 네트워크의 파라미터 $\theta$ 와 $w$ 초기화<br>
    * 2) 상태 $s$ 초기화<br>
    * 3) 액션 $a \sim \pi_\theta(a\mid s)$ 를 샘플링<br>
    * 4) 에피소드가 끝날때까지 A~E 반복<br>
      * A) $a$ 를 실행하여 보상 $r$ 과 다음 상태 $s'$ 을 얻음<br>
      * B) $\theta$ 업데이트 : $\theta\leftarrow \theta+\alpha\nabla_\theta log \pi_\theta(s,a)*Q_w(s,a)$<br>
      * C) 액션 $a' \sim \pi_\theta(a'\mid s')$ 를 샘플링<br>
      * D) $w$ 업데이트 : $w\leftarrow w+\beta(r+\gamma Q_w(s',a')-Q_w(s,a))\nabla_w Q_w(s,a)$<br>
      * E) $a\leftarrow a', s\leftarrow s'$<br>
  * 어드밴티지 액터-크리틱<br>
$A_{\pi_\theta}(s,a)\equiv Q_{\pi_\theta}(s,a)-V_{\pi_\theta}(s)$<br>
$A_{\pi_\theta}(s,a)$ : Advantage<br>
$V_{\pi_\theta}(s)$ : Baseline (기저)<br>
$E_{\pi_\theta}[\nabla_\theta log \pi_\theta(s,a)* B(s)]=\sum_{s\in S}d_{\pi_\theta}(s)\sum_{a\in A}\pi_\theta(s,a)\nabla_\theta log \pi_\theta(s,a)* B(s)$<br>
$\sum_{s\in S}d_{\pi_\theta}(s)\sum_{a\in A}\pi_\theta(s,a)\nabla_\theta log \pi_\theta(s,a)* B(s)=\sum_{s\in S}d_{\pi_\theta}(s)\sum_{a\in A}\pi_\theta(s,a)\frac{\nabla_\theta \pi_\theta(s,a)}{\pi_\theta(s,a)}* B(s)$<br>
$\sum_{s\in S}d_{\pi_\theta}(s)\sum_{a\in A}\pi_\theta(s,a)\nabla_\theta log \pi_\theta(s,a)* B(s)=\sum_{s\in S}d_{\pi_\theta}(s)\sum_{a\in A}\nabla_\theta \pi_\theta(s,a)* B(s)$<br>
$\sum_{s\in S}d_{\pi_\theta}(s)\sum_{a\in A}\pi_\theta(s,a)\nabla_\theta log \pi_\theta(s,a)* B(s)=\sum_{s\in S}d_{\pi_\theta}(s)B(s)\sum_{a\in A}\nabla_\theta \pi_\theta(s,a)$<br>
$\sum_{s\in S}d_{\pi_\theta}(s)\sum_{a\in A}\pi_\theta(s,a)\nabla_\theta log \pi_\theta(s,a)* B(s)=\sum_{s\in S}d_{\pi_\theta}(s)B(s)\nabla_\theta\sum_{a\in A} \pi_\theta(s,a)$<br>
$\sum_{s\in S}d_{\pi_\theta}(s)\sum_{a\in A}\pi_\theta(s,a)\nabla_\theta log \pi_\theta(s,a)* B(s)=\sum_{s\in S}d_{\pi_\theta}(s)B(s)\nabla_\theta1=0$<br>
$\therefore E_{\pi_\theta}[\nabla_\theta log \pi_\theta(s,a)* B(s)]=0$<br>
$\nabla_\theta J(\theta)=E_{\pi_\theta}[\nabla_\theta log \pi_\theta(s,a)*A_{\pi_\theta}(s,a)]$<br>
$A_{\pi_\theta}(s,a)=Q_{\pi_\theta}(s,a)-V_{\pi_\theta}(s)$<br>
$Q_{\pi_\theta}\approx Q_w$<br>
$V_{\pi_\theta}\approx V_\phi$<br>
    * 정책함수 $\pi_\theta(s,a)$ 의 뉴럴넷 $\theta$<br>
    * 액션-가치함수 $Q_w(s,a)$ 의 뉴럴넷 $w$<br>
    * 가치함수 $V_\phi$ 의 뉴럴넷 $\phi$<br>
  * 어드밴티지 액터-크리틱 Pseudo Code<br>
    * 1) 3쌍의 뉴럴넷 파라미터 $\theta,w,\phi$ 초기화<br>
    * 2) 상태 $s$ 초기화<br>
    * 3) 액션 $a \sim \pi_\theta(a\mid s)$ 를 샘플링<br>
    * 4) 에피소드가 끝날때까지 A~F 반복<br>
      * A) $a$ 를 실행하여 보상 $r$ 과 다음 상태 $s'$ 을 얻음<br>
      * B) $\theta$ 업데이트 : $\theta\leftarrow \theta+\alpha_1\nabla_\theta log \pi_\theta(s,a)* (Q_w(s,a)-V_\phi(s))$<br>
      * C) 액션 $a' \sim \pi_\theta(a'\mid s')$ 를 샘플링<br>
      * D) $w$ 업데이트 : $w\leftarrow w+\alpha_2(r+\gamma Q_w(s',a')-Q_w(s,a))\nabla_w Q_w(s,a)$<br>
      * E) $\phi$ 업데이트 : $\phi\leftarrow \phi+\alpha_3(r+\gamma V_\phi(s')-V_\phi(s))\nabla_\phi V_\phi(s)$<br>
      * F) $a\leftarrow a', s\leftarrow s'$<br>
  * TD 액터-크리틱<br>
$\delta=r+\gamma V(s')-V(s)$<br>
$E_\pi[\delta\mid s,a]=E_\pi[r+\gamma V(s')-V(s)\mid s,a]$<br>
$E_\pi[\delta\mid s,a]=E_\pi[r+\gamma V(s')\mid s,a]-V(s)$<br>
$E_\pi[\delta\mid s,a]=Q(s,a)-V(s)=A(s,a)$ : $\delta$ 는 $A(s,a)$ 의 불편추정량<br>
$\nabla_\theta J(\theta)=E_{\pi_\theta}[\nabla_\theta log \pi_\theta(s,a)*\delta]$<br>
    * 어드밴티지와 기저
    * 상태 분포 : 정책 $\theta$를 따라서 움직이는 에이전트가 각 상태에 평균적으로 머무는 비율을 나타내는 분포
    * 정책 함수, 액션-가치 함수, 가치 함수 3개 학습 필요
    * 그라디언트 추정치의 변동성을 줄여줌으로써 효율적인 학습 가능
  * TD Actor-Critic Pseudo Code<br>
    * 1) 정책, 밸류 네트워크 파라미터 $\theta,\phi$ 초기화<br>
    * 2) 액션 $a \sim \pi_\theta(a\mid s)$ 를 샘플링<br>
    * 3) 에피소드가 끝날때까지 A~E 반복<br>
      * A) $a$ 를 실행하여 보상 $r$ 과 다음 상태 $s'$ 을 얻음<br>
      * B) $\delta$ 계산 : $\delta\leftarrow r+\gamma V_\phi(s')-V_\phi(s)$<br>
      * C) $\theta$ 업데이트 : $\theta\leftarrow \theta+\alpha_1\nabla_\theta log \pi_\theta(s,a)*\delta$<br>
      * D) $\phi$ 업데이트 : $\phi\leftarrow \phi+\alpha_2\delta\nabla_\phi V_\phi(s)$<br>
      * E) $a\leftarrow a', s\leftarrow s'$<br>
  * TD Error의 기댓값이 어드밴티지 (TD Error는 어드밴티지의 불편 추정량)
  * TD Error를 이용함으로써 정책 함수, 가치 함수 2개만 학습하면 됨
  * [TD Actor-Critic 구현 사례](https://github.com/seungeunrho/RLfrombasics/blob/master/ch9_ActorCritic.py)
    * loss function 계산시, detach() 함수 반영사실에 주의
```
loss = -torch.log(pi_a) * delta.detach() + F.smooth_l1_loss(self.v(s), td_target.detach())
```
    * delta.detach()는 $\delta$ 를 상수처리 해서 업데이트 되지 않도록 하기 위함<br>
    * td_target.detach()는 TD Target을 상수처리 해서 업데이트 되지 않도록 하기 위함<br>
    * 정답은 그 자리에 가만히 있고, 예측치가 변하도록 하기 위함<br>
    * Loss는 정책 네트워크의 손실함수와 밸류 네트워크의 손실함수를 더해서 계산
  * Policy Gradient 알고리즘<br>
$\nabla_\theta J(\theta)=E_{\pi_\theta}[\nabla_\theta log \pi_\theta(s,a)* Q_{\pi_\theta}(s,a)]$ : Policy Gradient Theorem<br>
$\nabla_\theta J(\theta)=E_{\pi_\theta}[\nabla_\theta log \pi_\theta(s,a)* G_t]$ : REINFORCE<br>
$\nabla_\theta J(\theta)=E_{\pi_\theta}[\nabla_\theta log \pi_\theta(s,a)* Q_w(s,a)]$ : Q Actor Critic<br>
$\nabla_\theta J(\theta)=E_{\pi_\theta}[\nabla_\theta log \pi_\theta(s,a)* A_w(s,a)]$ : Advantage Actor Critic<br>
$\nabla_\theta J(\theta)=E_{\pi_\theta}[\nabla_\theta log \pi_\theta(s,a)* \delta]$ : TD Actor Critic<br>



# 10. 알파고와 MCTS
* 10.1 알파고
  * 학습 : 이세돌과 경기전 기보 기반 학습
  * 실시간 플래닝 (MCTS) : 이세돌과 경기중 실시간 플래닝
  * 학습 단계<br>
    * 지도학습 : $\pi_{sl}, \pi_{roll}$<br>
$\pi_{sl}$ : 컨볼루션 레이어 (13층), Input Feature (19 * 19 * 48), Output (19 * 19)<br>
$\pi_{roll}$ : 선형결합 레이어 (1층), Input Feature (19 * 19 * 48), Output (19 * 19), $\pi_{sl}$ 의 가벼운 버전<br>
    * 강화학습 : $\pi_{rl}, v$<br>
$\pi_{rl}$ : $\pi_{sl}$ 과 동일, 자신의 과거 모델들과 Self-Play 진행<br>
$\nabla_\theta J(\theta)=E_{\pi_\theta}[\nabla_\theta log \pi_\theta(s,a)* G_t]$ : 보상함수에 REINFORCE 사용<br>
$v_{\pi_{rl}}(s)=E_{\pi_{rl}}[G_t\mid s_t=s]$<br>
  * MCTS
    * 선택 > 확장 > 시뮬레이션 > 백프로파게이션 반복 진행
    * 선택 : 루트 노드에서 출발하여 리프 노드까지 가는 단계<br>
$a_t=argmax_a(Q(s_t,a)+u(s_t,a))$ : 경험이 쌓일수록 $Q(s,a)$ 의 영향력은 커지고, $u(s,a)$ 의 영향력은 작아짐<br>
$Q(s_t,a)$ : 시뮬레이션 실행 후 얼마나 좋은지 판단<br>
상태 $s_{78}$ 에서 액션 $a_{33}$ 을 선택하는 경험을 총 100번 경험했다면 각 경험마다 리프 노드에 도달할 것이고, 해당 리프 노드가 $s_L^1,\cdots,s_L^{100}$ 일 경우<br>
$Q(s_{78},a_{33})=\frac{1}{100}\sum_{i=1}^{100}V(s_L^i)$<br>
$u(s_t,a)$ : 시뮬레이션 실행 전 얼마나 좋을 것이라 추측하는지 판단(Prior)<br>
$u(s_t,a)\varpropto \frac{P(s,a)}{1+N(s,a)}$<br>
$P(s,a)=\pi_{sl}(s,a)$ : 사전 확률 (Prior Probability), 시뮬레이션 해보기 전에 각 액션에 확률 부여<br>
    * 확장 : 리프 노드를 실제 트리의 노드로 확장하는 단계<br>
$P(s,a)\leftarrow \pi_{sl}(s,a)$<br>
$N(s,a)\leftarrow 0$<br>
$Q(s,a)\leftarrow 0$<br>
    * 시뮬레이션 : 정식 노드가 된 리프 노드의 가치를 평가하는 단계<br>
리프 노드 $s_L$ 부터 시작해서 게임이 끝날때까지 빠르게 시뮬레이션 ($\pi_{roll}$ 활용)하고, 그 결과인 $z_L$ 을 $s_L$ 의 밸류로 활용<br>
밸류 네트워크 $v_{rl}(s_L)$ 을 활용<br>
$V(s_L)=\frac{v_{rl}(s_L)+z_L}{2}$ 을 최종 밸류로 결정<br>
    * 백프로파게이션 : 리프 노드에 도달하기까지 지나온 모든 엣지에 대해 $Q(s,a)$ 와 $N(s,a)$ 를 업데이트하는 단계<br>
$N(s,a)\leftarrow N(s,a)+1$<br>
$Q(s,a)\leftarrow Q(s,a)+\frac{1}{N(s,a)}(V(s_L)-Q(s,a))$<br>
    * 가장 좋은 액션을 고르는 기준? $N(s,a)$ 가 가장 큰 액션
      * 신뢰도를 함께 고려하기 위함

* 10.2 알파고 제로
  * MCTS : 현재 상태를 인풋으로 받아서 그에 특화된 정책을 내놓는 모듈<br>
    * 각각의 데이터 : $(s_t,\pi_t,z_t)$ 로 구성<br>
$s_t$ : $t$ 시점의 상태<br>
$\pi_t$ : $s_t$ 에서 진행한 MCTS가 알려주는 정답 정책<br>
$z_t$ : 게임 결과값<br>
    * 위 데이터를 이용해 뉴럴넷 $f_\theta$ 학습<br>
$MCTS(s)=(\pi,z)$<br>
$f_\theta(s)=(p,v)$<br>
$p\approx \pi$<br>
$v\approx z$<br>
    * 학습 프로세스<br>
      * 1) 상태 $s_t$ 를 $f_\theta$ 에 인풋으로 넣어서 정책 네트워크 아웃풋 $p_t$ 와 밸류 네트워크 아웃풋 $v_t$ 를 계산<br>
      * 2) $p_t$ 는 MCTS가 알려주는 확률 분포인 $\pi_t$ 와의 차이를 줄이는 방향으로 업데이트<br>
      * 3) $v_t$ 는 경기 결과값인 $z_t$ 와의 차이를 줄이는 방향으로 업데이트<br>
    * 손실함수<br>
$L(\theta)=(z_t-v_t)^2-\pi_t log(p_t)$<br>
  * 알파고 제로에서의 MCTS
    * 선택 > 확장 및 평가 > 백프로파게이션 반복 진행<br>
$\pi_{sl}$ 대신 $p$ 사용, $v_{rl}$ 대신 $v$ 사용, $\pi_{roll}$ 은 사용하지 않음<br>
    * 확장 단계 : $f_\theta$ 아웃풋인 $p$ 이용해 초기화 진행<br>
    * 평가 단계 : $f_\theta$ 아웃풋인 $v$ 이용해 $s_L$ 밸류 평가<br>
    * 랜덤 정책을 이용해서 MCTS를 해보고, 그 중에서 결과가 좋았던 액션을 선택 (MCTS가 선생님 역할 수행)<br>
    * MCTS 덕분에 뉴럴넷 $f_\theta$ 의 아웃풋 $p, v$ 도 점점 더 정확해짐<br>



# 11. 블레이드 & 소울 비무 AI 만들기
* 11.1 블레이드 & 소울 비무
  * 거대한 문제 공간
  * 실시간 게임이 갖는 제약 (cf. 턴제 게임)
  * 물고 물리는 스킬 관계
  * 상대방에 관계없는 Robust한 성능
* 11.2 비무에 강화학습 적용하기
  * MDP 만들기<br>
    * 관측치 ($o_t$) : $o_t,o_{t-1},o_{t-2},\cdots,o_1$ 를 이용해 상태 $s_t$ 정의 (RNN 기법인 LSTM 이용)<br>
    * 액션 ($a_t$) : $a_{skill}, a_{move,target}$ 도입<br>
    * 보상 ($r_t$) : $r_t=r_t^{WIN}+r_t^{HP}$ 정의 (Optimality와 Frequency 고려)<br>
$r_t^{HP}=(HP_t^{ag}-HP_{t-1}^{ag})-(HP_t^{op}-HP_{t-1}^{op})$ : 나와 적의 체력 차이<br>
  * 학습 시스템과 알고리즘
    * ACER (Actor-Critic with Experience Replay) 활용 : A3C의 Off-Policy 버전 알고리즘<br>
    * 학습대상 네트워크 : $\pi_{skill}, \pi_{move,target}, Q_{skill}, Q_{move,target}$<br>
    * 여러개의 정책 네트워크를 학습하는 방법<br>
      * 1) 번갈아가며 업데이트하는 방법 : $\pi_{skill}$ 업데이트 후 $\pi_{move,target}$ 업데이트<br>
      * 2) 서로 독립을 가정하고 곱셈을 이용해 계산하는 방법 : $\pi(a_{skill},a_{move,target}\mid s)=\pi_{skill}(a_{skill}\mid s)* \pi_{move,target}(a_{move,target}\mid s)$<br>
      * 3) 하나씩 순차적으로 선택하는 방법 : $\pi(a_{skill},a_{move,target}\mid s)=\pi_{skill}(a_{skill}\mid s)* \pi_{move,target}(a_{move,target}\mid s,a_{skill})$<br>
    * 이동 정책 네트워크의 학습 : 한 틱 (0.1초) 동안 움직이는 거리가 매우 제한적이라 한 번 이동 방향이 결정되면, 이후 9틱 동안 같은 방향으로 움직이도록 제한
* 11.3 전투 스타일 유도를 통한 새로운 방식의 Self-Play 학습
  * 보상을 통한 전투 스타일 유도 (보상 조절 방법)
    * HP 비율, 시간, 거리 패널티 설정을 통해 공격형, 수비형, 밸런스형 스타일 학습
  * 새로운 Self-Play 커리큘럼
    * 3가지 스타일의 에이전트가 공통의 풀에서 다양한 스타일의 대전 상대를 만날 수 있도록 설정



# 참고자료
* [https://github.com/seungeunrho/RLfrombasics](https://github.com/seungeunrho/RLfrombasics)
