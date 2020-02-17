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

export PATH=/usr/local/bin:/usr/local/sbin:${PATH}
\# Setting PATH for Python 3.7
\# The original version is saved in .bash_profile.pysave
PATH="/Library/Frameworks/Python.framework/Versions/3.7/bin:${PATH}"
export PATH
{: .notice--info}

* Console 실행 명령
  * python3
  * pip3

* [PyCharm](https://www.jetbrains.com/pycharm/)
  * 2019.3 버전 설치
  * 신규 프로젝트 생성 : ~/PyCharmProjects/MF_Python/
  * Preferences > Project: TensorFlow > Project Interpreter > Show All > + 표시
    * Location : /Users/kimsanghun/PycharmProjects/
    * Base Interpreter : /usr/local/bin/python3.7
    * Inherit global site-packages 체크 (Console상에서 설치한 패키지 그대로 상속)
  * 하단 패키지 목록 확인
  * + 표시 눌러서 패키지 추가 가능하나 Console에서 추가해야 함
    * PyCharm에서 추가시 Console에서 사용 불가!

---
