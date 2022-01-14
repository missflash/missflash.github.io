---
title: "Python Setup on Mac"
date: 2020. 2. 16. 오후 4:47:00
categories:
use_math: false
classes: wide
---

* [Python](https://www.python.org/downloads/release/python-376/)
  * 3.7.6 버전 설치
  * /usr/local/bin 폴더
  * /Library/Frameworks/Python.framework 폴더
  * /Applications/Python 3.7 폴더

* Path 추가
  * vi ~/.bash_profile

export PATH=/usr/local/bin:/usr/local/sbin:${PATH}<br>
\# Setting PATH for Python 3.7<br>
\# The original version is saved in .bash_profile.pysave<br>
PATH="/Library/Frameworks/Python.framework/Versions/3.7/bin:${PATH}"<br>
export PATH
{: .notice--info}

* Console 실행 명령
  * python3
  * pip3

* [PyCharm](https://www.jetbrains.com/pycharm/)
  * 2019.3 버전 설치
  * 신규 프로젝트 생성 : ~/PyCharmProjects/MF_JSSP/
  * Preferences > Project: TensorFlow > Project Interpreter > Show All > + 표시
    * Location : /Users/kimsanghun/PycharmProjects/MF_JSSP
    * Base Interpreter : /usr/local/bin/python3.7
    * Inherit global site-packages 체크 (Console상에서 설치한 패키지 그대로 상속)
  * 하단 패키지 목록 확인
  * \+ 표시 눌러서 패키지 추가 가능하나 Console에서 추가해야 함
    * PyCharm에서 추가시 Console에서 사용 불가!
  * JSSP 필요 패키지
    * pyjssp
    * torch
      * pip install pytorch==1.8.0+cpu torchvision==0.9.0+cpu torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html
      * pip install torch-scatter torch-sparse torch-cluster -f https://pytorch-geometric.com/whl/torch-1.8.0+cpu.html
      * pip install torch-geometric
    * torch-scatter
    * torch-sparse
    * torch-cluster
    * torch-geometric
  * requirements.txt 파일 있을 경우, 아래와 같이 필요 라이브러리 Install 가능
    * pip install --upgrade -r requirements.txt
      * 전체 패키지 버전 확인 : pip list
      * 패키지 버전 확인 : pip show tensorflow
      * 특정 버전 패키지 업데이트 : pip install scikit-learn==0.22.1
  * PyCharm에서 가상환경 변경
    * 설정 > Project > Project Interpreter
  * 터미널에서 가상환경 변경
    * 가상환경 활성화 : source ~/PycharmProjects/MF_JSSP/bin/activate
    * 가상환경 종료 : deactivate

* Python 환경설정 관련
  * Python 설치 경로 확인 : where python
  * 설치 경로별 pip 실행 : /Python 설치 경로/bin/python -m pip install 패키지명
  * 카카오 Repository pip 실행 : /Python 설치 경로/bin/python -m pip install -i http://ftp.daumkakao.com/pypi/simple --trusted-host ftp.daumkakao.com 패키지명

---
