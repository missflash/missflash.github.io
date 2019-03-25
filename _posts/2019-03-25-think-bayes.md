---
title: "Think Bayes"
date: 2019. 3. 25. 오후 9:23:22
categories:
use_math: true
---

# 1. 베이즈 이론
$p(A|B) = \frac {p(A)p(B|A)}{p(B)}$<br>
$p(A)$ : 사전 확률<br>
$p(A|B)$ : 사후 확률<br>
$p(B|A)$ : 우도<br>
$p(B)$ : 한정 상수<br>

M&M 문제

몬티 홀 문제


# 2. 계산 통계
[베이지안 프레임워크](http://thinkbayes.com/monty2.py) 사용 방법 설명


# 3. 추정 1
사전 확률을 높이는 방법
* 데이터를 더 확보할 것
* 배경 지식을 더 확보할 것

신뢰구간

누적 분포 함수


# 4. 추정 2
사전 분포 범람 : 데이터가 충분하다면 서로 다른 사전 확률을 가지고도 동일한 사후 확률로 수렴하는 경향이 있음

베타 분포


# 5. 공산과 가산
5.1 공산 : 내가 건 말이 이길 확률이 10%라면, 공산은 9:1으로 표현

5.2 베이즈 이론의 공산 형태 수식에서 Typo 확인
$\frac {p(A|D)}{p(B|D)} = \frac {p(H)p(D|H)}{p(B)P(D|B)}$<br>
위 수식 좌변 분자가 $p(A)p(D|A)$로 수정되어야 함

5.4 가산 분포 계산 방법
* 시뮬레이션
* 나열

5.5 최댓값 분포 계산 방법
* 시뮬레이션
* 나열
* 멱법

5.6 혼합 분포


# 6. 의사 결정 분석
* 확률 밀도 함수 (PDF)
* 커널 밀도 추정 (KDE) : 샘플로 데이터에 적합한 추정 평활 PDF 탐색)


# 7. 예측
* 포아송 프로세스
  * 프로세스 : 물리 시스템에 대한 추계 모델
  * 추계 : 모델에 몇 가지 임의성이 포함
* [http://thinkbayes.com/hockey.py](http://thinkbayes.com/hockey.py)


# 8. 관측 편향
* 관측자 편향 사례
  * 학생들의 대부분은 큰 강의실에 있기 때문에 강의실들이 실제보다 더 크다고 생각
  * 항공기 승객들은 보통 만석인 비행기에 타기 때문에 비행기가 실제보다 만석인 경우가 많다고 생각
* 레드라인 문제
  * 모델
  * 대기시간
  * 대기시간 예측
  * 도착 비율 추정
  * 결합 불확실성
    * 불확실한 변수의 결정값을 기반으로 한 분석을 구현
    * 불확실 변수값의 분포 계산
    * 변수의 각 값에 대해 분석을 실행한 후 예측 분포 셋 생성
    * 변수의 분포에 가중치를 줘서 예측 분포의 혼합을 계산
  * 의사 결정 분석
* [http://thinkbyaes.com/redline.py](http://thinkbyaes.com/redline.py)


# 9. 두 차원
* 페인트볼 게임
* 스윗
* 삼각법
* 우도
* 결합 분포
  * 다차원 공간에서 각 가능한 값과 이 값에 대한 확률을 나타내는 분포
* 주변 분포
  * 다른 변수를 모르는 상태로 둘 때 결합 분포 안에 있는 한 변수의 분포
* 조건 분포
  * 결합 분포 내에서 한 개 이상의 다른 변수의 상태에 따른 한 변수의 분포
* 신뢰구간
  * 중심 신뢰구간


# 10. 근사 베이지안 계산


# 11. 가설 검정


# 12. 증거


# 13. 시뮬레이션


# 14. 계층 모델


# 15. 차원 다루기


# 참고자료
* [https://github.com/AllenDowney/ThinkBayes](https://github.com/AllenDowney/ThinkBayes)
* [http://thinkbayes.com/thinkbayes.py](http://thinkbayes.com/thinkbayes.py)
* [http://thinkbayes.com/thinkbayes_code.zip](http://thinkbayes.com/thinkbayes_code.zip)

***

이하는 여담, 간만에 빠른 의식의 흐름으로 전개...

* 5.3 올리버의 혈액형 설명이 조금 이상해서 구글 검색
* [John Grib 님의 친절한 설명](https://johngrib.github.io/wiki/Oliver-s-blood/) 확인
* 관련글에서 [나의 공부 방법](https://johngrib.github.io/wiki/my-study-method/) 확인
* 체계적인 정리 방식에 매료되서 [정리 환경](https://johngrib.github.io/wiki/my-wiki/) 설정 참조
* 이렇게까지 신경쓸 여력은 없을 것 같아 [Dreamgonfly 님의 글](https://dreamgonfly.github.io/2018/01/27/jekyll-remote-theme.html) 참조 후 Github.io 블로그 개설
* 생각의 정리는 위키가 좋을 것 같아 Github Repository내 Wiki 사용 결정
* [Markdown 사용법](https://gist.github.com/ihoneymon/652be052a0727ad59601) 숙지 후 첫 페이지 작성
* 수식 추가를 위해 MathJax 추가 방법 검색 ([Minki Kim님](https://mkkim85.github.io/blog-apply-mathjax-to-jekyll-and-github-pages/), [jamiekang님](https://jamiekang.github.io/2017/04/28/blogging-on-github-with-jekyll/))했으나, Remote Theme 환경에서는 제대로 반영 안되는 문제 발생 ~~(해결방법 찾는중)~~ - [Minki Kim님](https://mkkim85.github.io/blog-apply-mathjax-to-jekyll-and-github-pages/) 글 정독 후 블로그는 성공했으나, Wiki는 여전히 적용안됨
* Wiki에서는 생각보다 안되는 제약이 많음 (Mathjax, [Gist](https://gist.github.com/missflash/) 등)
