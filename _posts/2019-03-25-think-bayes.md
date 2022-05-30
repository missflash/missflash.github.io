---
title: "Think Bayes"
date: 2019. 3. 25. 오후 9:23:22
categories:
use_math: true
toc: true
toc_label: "Table of Contents"
toc_icon: "cog"
toc_sticky: true
---

![Think Bayes](https://raw.githubusercontent.com/missflash/missflash.github.io/master/_files/think_bayes.jpg){: width="30%" height="30%"}


# 1. 베이즈 이론
* 베이즈 이론
$p(A|B) = \frac {p(A)p(B|A)}{p(B)}$<br>
$p(A)$ : 사전 확률<br>
$p(A|B)$ : 사후 확률<br>
$p(B|A)$ : 우도<br>
$p(B)$ : 한정 상수<br>
* M&M 문제
* 몬티 홀 문제



# 2. 계산 통계
[베이지안 프레임워크](http://thinkbayes.com/monty2.py) 사용 방법 설명



# 3. 추정 1
* 사전 확률을 높이는 방법
  * 데이터를 더 확보할 것
  * 배경 지식을 더 확보할 것
* 신뢰구간
* 누적 분포 함수



# 4. 추정 2
* 사전 분포 범람 : 데이터가 충분하다면 서로 다른 사전 확률을 가지고도 동일한 사후 확률로 수렴하는 경향이 있음
* 베타 분포



# 5. 공산과 가산
5.1 공산 : 내가 건 말이 이길 확률이 10%라면, 공산은 9:1으로 표현

5.2 베이즈 이론의 공산 형태 수식에서 Typo 확인
$\frac {p(A|D)}{p(B|D)} = \frac {p(H)p(D|H)}{p(B)P(D|B)}$<br>
위 수식 우변 분자가 $p(A)p(D|A)$로 수정되어야 함

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
* 평균과 표준편차
  * 분포의 평균이 $\mu$, 표준편차가 $\sigma$ 일 때, $n$개의 샘플을 취하면...
  * $\mu$의 추정치는 샘플 평균인 $m$
  * $\sigma$의 추정치는 샘플 표준편차인 $s$
  * 추정된 $\mu$의 표준편차는 $\frac{s}{\sqrt x}$
  * 추정된 $\sigma$의 표준 오차는 $\frac{s}{\sqrt {2(n-1)}}$
* 언더플로
  * 데이터셋의 크기가 커질수록 소수점 이하를 곱한 결과가 너무 작아 0으로 수렴
  * 해법1) 매번 갱신후 혹은 100개를 다시 돌려서 Pmf를 다시 정규화
  * 해법2) 로그를 이용해 곱하기를 더하기로 변환
* 근사 베이지안 계산
  * 사용1) 특히 큰 데이터셋의 경우, 매우 작아서 로그 변환이 어려울 때
  * 사용2) 계산 비용이 많이 들어서 최적화를 많이 해야하는 경우
  * 원값보다 빠르게 계산하기 위해 사용하는 추정값
  * 주어진 가설하에서 데이터의 우도는 어떻게 돼? 라는 질문에 대한 대답, 우도의 대안 모델
* 로버스트 추정
  * Outlier 제거로 데이터 신뢰도 향상



# 11. 가설 검정
* 베이즈 요인
  * 1~3 : 언급할 가치도 없음 (Barely Worth Mentioning)
  * 3~10 : 적당함 (Substantial)
  * 10~30 : 강력함 (Strong)
  * 30~100 : 매우 강력함 (Very Strong)
  * 100 이상 : 확실함 (Decisive)



# 12. 증거
* 사전 분포
* 사후 분포
  * 가설 A : p_correct는 밥보다 앨리스가 더 높다
  * 가설 B : p_correct는 앨리스보다 밥이 더 높다
  * A, B의 우도 계산한 뒤 우도비 (베이즈 요인) 확인
* 더 나은 모델
  * 모든 시험 응시자가 동일한 효과 가정
* 보정
* 효과의 사후 분포
* 예측 분포



# 13. 시뮬레이션
* 신장 종양 문제
* 단순 모델
  * 종양의 배가 시간은 일정한 상수
  * 최대 한 쪽 길이가 두 배가 되면, 부피는 8배가 되는 3차원 형태
* 좀 더 일반적인 모델
  * 종양 성장 시뮬레이션을 통해 종양 나이별 크기 상태의 분포 생성
* 구현
* 결합 확률 캐싱
  * 의사코드 (Pseudocolor) 활용해 결합 분포 표현
* 조건 분포
  * 크기별 종양 나이 분포
* 연속 상관관계
  * 연속 상관관계가 높은 종양은 시간이 지날수록 크기가 더 커짐



# 14. 계층 모델
* 가이거 계수기 문제
  * 정문제 (Forward Problem) : 시스템 변수가 주어지면 데이터의 분포를 알 수 있음
  * 역문제 (Inverse Problem) : 데이터가 주어지면 변수의 분포를 알 수 있음
* 단순하게 시작하기
* 계층적으로 만들기
  * 메타 스윗 (Suite) : 다른 스윗을 값으로 가지는 것
  * 계층적 : 스윗이 여러 단계로 사용되는 것
* 약간 최적화하기
* 사후 분포 추출하기



# 15. 차원 다루기
* 배꼽 박테리아
  * 보이지 않는 종 문제 (Unseen Species Problem)
* 사자와 호랑이와 곰
* 계층 버전
* 랜덤 샘플링
  * 디리클레 분포에서 임의의 샘플을 만드는 방법
    * 주변 베타 분포를 사용해 한 번에 하나만 선택하고 나머지를 더해서 1이 되도록 맞추는 방법
    * n개의 감마 분포에서 값을 선택해서 총 합으로 이를 나눠서 정규화 하는 방법
* 최적화
  * 최적화 첫 단계는 동일한 데이터로 디리클레 분포를 갱신할 때 처음 m개의 변수는 전체에 대해서 동일하다는 것을 깨닫는 것이다. 유일한 차이점이라면 가설에서는 보이지 않는 종의 숫자다. 따라서 실제로는 n개의 디리클레 객체가 필요한 것이 아니기 때문에, 계층의 최상위 객체에 이 변수를 저장할 수 있다. (Species2 모델)
  * Species2도 가설 전체에 대해 임의의 값에 대한 동일한 집합을 사용한다. 이를 통해 임의의 값을 생성하는 시간이 단축되지만, 이보다 더 중요한 이점이 있으므로 이는 두 번째 이점에 불과하다. 바로 전체 가설을 샘플 공간에서 동일하게 선택하게 함으로써 가설을 보다 공평하게 비교할 수 있으므로 결과를 수렴하는데 더 적은 횟수의 반복만 하면 되는 것이다. (Species3 모델)
  * 이런 변화에도 불구하고 가장 중요한 성능 문제가 있다. 관측되는 종의 수가 증가할수록 임의의 확산도에 대한 배열은 커지고 이 중 하나가 선택될 확률은 작아진다. 따라서 대부분의 수많은 반복은 전체 합에 그다지 도움되지 않는 매우 작은 우도만을 만들고 가설간에 차별점도 생기지 않을 것이다.
  * 이에 대한 해결책으로는 한 번에 한 종씩 갱신하는 것이다. Species4는 하위 가설을 나타내는 디리클레 객체를 사용해서 이 방법을 간단히 구현한 것이다.
  * 마지막으로 Species5는 최상위 레벨에서 하위 가설을 결합하고 numpy 배열 연산자를 이용해 속도를 높인다.
* 배꼽 박테리아 데이터
* 예측 분포
* 결합 사후 분포
* 범위



# 참고자료
* [https://github.com/AllenDowney/ThinkBayes](https://github.com/AllenDowney/ThinkBayes)
* [http://thinkbayes.com/thinkbayes.py](http://thinkbayes.com/thinkbayes.py)
* [http://thinkbayes.com/thinkbayes_code.zip](http://thinkbayes.com/thinkbayes_code.zip)

***

<a id="More"></a>이하는 여담, 간만에 빠른 의식의 흐름으로 전개...

* 5.3 올리버의 혈액형 설명이 조금 이상해서 구글 검색
* [John Grib 님의 친절한 설명](https://johngrib.github.io/wiki/Oliver-s-blood/) 확인
* 관련글에서 [나의 공부 방법](https://johngrib.github.io/wiki/my-study-method/) 확인
* 체계적인 정리 방식에 매료되서 [정리 환경](https://johngrib.github.io/wiki/my-wiki/) 설정 참조
* 이렇게까지 신경쓸 여력은 없을 것 같아 [Dreamgonfly 님의 글](https://dreamgonfly.github.io/2018/01/27/jekyll-remote-theme.html), [공식 문서](https://mmistakes.github.io/minimal-mistakes/docs/quick-start-guide/#remote-theme-method) 참조 후 Github.io 블로그 개설
* 생각의 정리는 위키가 좋을 것 같아 Github Repository내 Wiki 사용 결정
* [Markdown 사용법](https://gist.github.com/ihoneymon/652be052a0727ad59601) 숙지 후 첫 페이지 작성
* 수식 추가를 위해 MathJax 추가 방법 검색 ([Minki Kim님](https://mkkim85.github.io/blog-apply-mathjax-to-jekyll-and-github-pages/), [jamiekang님](https://jamiekang.github.io/2017/04/28/blogging-on-github-with-jekyll/))했으나, Remote Theme 환경에서는 제대로 반영 안되는 문제 발생 ~~(해결방법 찾는중)~~ - [Minki Kim님](https://mkkim85.github.io/blog-apply-mathjax-to-jekyll-and-github-pages/) 글 정독 후 블로그는 성공했으나, Wiki는 여전히 적용안됨
* Wiki에서는 생각보다 안되는 제약이 많음 (Mathjax, [Gist](https://gist.github.com/missflash/) 등)
* [Blog 설정](https://mmistakes.github.io/minimal-mistakes/docs/configuration/) 확인
* [취미로 코딩하는 개발자](https://devinlife.com/howto/)에서 기타 블로그 설정 참조
