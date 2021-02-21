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
$MP≡(S,P)$<br>
  * 마르코프 프로세스 구성요소
    * 상태의 집합 $S$<br>
    * 전이 확률 행렬 $P_{SS'}$<br>
  * 마르코프 성질<br>
$P[S_{t+1}|S_t]=P[S_{t+1}|S_1,S_2,\cdots,S_t]$
* 2.2 마르코프 리워드 프로세스 (Markov Reward Process)
  * 마르코프 리워드 프로세스 정의
    * 마르코프 프로세스에 보상의 개념이 추가<br>
$MRP≡(S,P,R,\gamma)$<br>
  * 마르코프 리워드 프로세스 구성요소
    * 상태의 집합 $S$<br>
    * 전이 확률 행렬 $P_{SS'}$<br>
    * 보상함수 $R=E[R_t|S_t=s]$<br>
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
$MDP≡(S,A,P,R,\gamma)$<br>
  * 마르코프 결정 프로세스 구성요소
    * 상태의 집합 $S$<br>
    * 액션의 집합 $A$<br>
    * 전이 확률 행렬 $P^a_{SS'}=P[S_{t+1}=s'|S_t=s,A_t=a]$<br>
    * 보상함수 $R^a_s=E[R_{t+1}|S_t=s,A_t=a]$<br>
    * 감쇠인자 $\gamma$<br>
  * 정책합수와 2가지 가치함수
    * 정책함수<br>
$\pi(a|s)=P[A_t=a|S_t=s]$<br>
    * 상태 가치함수<br>
$v_\pi(s)=E_\pi[r_{t+1}+\gamma r_{t+2}+\gamma^2 r_{t+3}+\cdots|S_t=s]\\ ~~~~~~~~=E_\pi[G_t|S_t=s]$<br>
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
$v_\pi(s_t)=\sum_{a\in A}\pi(a|s)q_\pi(s,a)$<br>
$q_\pi(s_t,a_t)=r^a_s+\gamma \sum_{s'\in S}P^a_{ss'}v_\pi(s')$<br>
  * 2단계<br>
$v_\pi(s_t)=\sum_{a\in A}\pi(a|s)\left(r^a_s+\gamma \sum_{s'\in S}P^a_{ss'}v_\pi(s')\right)$<br>
$q_\pi(s_t,a_t)=r^a_s+\gamma \sum_{s'\in S}P^a_{ss'}\sum_{a'\in A}\pi(a'|s')q_\pi(s',a')$<br>

* 3.2 벨만 최적 방정식



# 4. MDP를 알 때의 플래닝
* 4.1 밸류 평가하기 - 반복적 정책 평가
* 4.2 최고의 정책 찾기 - 정책 이터레이션
* 4.3 최고의 정책 찾기 - 밸류 이터레이션



# 5. MDP를 모를 때 밸류 평가하기
* 5.1 몬테카를로 학습
* 5.2 Temporal Difference 학습
* 5.3 몬테카를로 vs TD
* 5.4 몬테카를로와 TD의 중간?



# 6. MDP를 모를 때 최고의 정책 찾기
* 6.1 몬테카를로 컨트롤
* 6.2 TD 컨트롤 1 - SARSA
* 6.3 TD 컨트롤 2 - Q러닝



# 7. Deep RL 첫 걸음
* 7.1 함수를 활용한 근사
* 7.2 인공 신경망의 도입



# 8. 가치 기반 에이전트
* 8.1 밸류 네트워크의 학습
* 8.2 딥 Q러닝



# 9. 정책 기반 에이전트
* 9.1 Policy Gradient
* 9.2 REINFORCE 알고리즘
* 9.3 액터-크리틱



# 10. 알파고와 MCTS
* 10.1 알파고
* 10.2 알파고 제로



# 11. 블레이드 & 소울 비무 AI 만들기
* 11.1 블레이드 & 소울 비무
* 11.2 비무에 강화학습 적용하기
* 11.3 전투 스타일 유도를 통한 새로운 방식의 Self-Play 학습



# 참고자료
* [https://github.com/seungeunrho/RLfrombasics](https://github.com/seungeunrho/RLfrombasics)
