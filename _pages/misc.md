---
title: "Personal Stuff."
permalink: /misc/
layout: single
author_profile: true
---

# Useful Codes
* [본문 폰트크기 변경](https://github.com/missflash/missflash.github.io/commit/273d4b95a962c96d531974ba378b272666dc6824)
* [Page 추가](https://github.com/missflash/missflash.github.io/commit/126a484a364cc69c44785f341d617a68620d8706)
  * [Others](https://github.com/mmistakes/minimal-mistakes/tree/master/docs/_pages)
* [Navigation 메뉴 추가](https://github.com/missflash/missflash.github.io/commit/39267d309f3adb76be11be2be28036c9d64f7574)
* Anchor 추가
  * [다음](/think-bayes/#More)

```
[다음](/think-bayes/#More)
...
<a id="More"></a>이하는 여담, 간만에 빠른 의식의 흐름으로 전개...
```

* Post 설정 예시

```
---
title: "Think Bayes"
date: 2019. 3. 25. 오후 9:23:22
categories:
tags:
use_math: true
toc: true
toc_label: "Table of Contents"
toc_icon: "cog"
toc_sticky: true
---
```

```
---
title: "Bayesian Methods for Hackers"
date: 2019. 4. 2. 오후 10:45:33
categories:
tags:
use_math: true
classes: wide
---
```

  * 비공개 전환 필요시 `published : false` 추가

* Page 설정 예시

```
---
title: "Personal Stuff."
permalink: /misc/
layout: single
author_profile: true
---
```

```
---
title: "Page Not Found"
excerpt: "Page not found. Your pixels are in another canvas."
permalink: /404.html
layout: single
author_profile: true
---
```

* 문장 강조

```
문장 강조
{: .notice--info}
```

문장 강조
{: .notice--info}

* Inline 강조

```
`Inline 강조` 예시
```

`Inline 강조` 예시

---

# github.io
* [설정 참고1](https://mmistakes.github.io/minimal-mistakes/docs/configuration/)
* [설정 참고2](https://devinlife.com/howto/)

---

# Markdown
* [문법 참고1](https://seoulrain.net/2014/12/03/writemonkey05/)
* [문법 참고2](http://taewan.kim/post/markdown/)
* Markdown을 Html로 변환하기
  * [Pandoc 설치](https://pandoc.org/installing.html)
  * 터미널 실행
  * pandoc 2020-01-30-palo-alto-log.md -f markdown -t html -s -o 2020-01-30-palo-alto-log.html

---

# MathJax
* [문법 참고](http://www.onemathematicalcat.org/MathJaxDocumentation/MathJaxKorean/TeXSyntax_ko.html)
* [실시간 확인](https://cdn.rawgit.com/mathjax/MathJax/2.7.1/test/sample-dynamic-2.html)

---

# 실행 스크립트
* github_push.sh (sh github_push.sh "Modify-Post.")

\#!/bin/sh<br>
cd /Users/kimsanghun/MissFlash/Github<br>
git remote add missflash https://github.com/missflash/missflash.github.io.git<br>
git remote -v<br>
git pull missflash master<br>
git add .<br>
git commit -m "$@"<br>
git push -u missflash master<br>
<br>
cd /Users/kimsanghun/MissFlash/Github_MF_Stuff<br>
git remote add mf_stuff https://github.com/missflash/MF_Stuff.git<br>
git remote -v<br>
git pull mf_stuff master<br>
git add .<br>
git commit -m "$@"<br>
git push -u mf_stuff master<br>
{: .notice--info}

* github_push2.sh (sh github_push2.sh "Update-Repository.")

\#!/bin/sh<br>
cd /Users/kimsanghun/Dropbox/MissFlash/Personal/Visiting_Scholars/4.Research/Samsung-KAIST<br>
git remote add Reinforcement_Project https://github.com/missflash/Reinforcement_Project.git<br>
git remote -v<br>
git pull Reinforcement_Project master<br>
git add .<br>
git commit -m "$@"<br>
git push -u Reinforcement_Project master<br><br>
cd /Users/kimsanghun/PycharmProjects/TensorFlowV1/lib/python3.7/site-packages/pyjssp<br>
git remote add pyjssp https://github.com/missflash/pyjssp.git<br>
git remote -v<br>
git pull pyjssp master<br>
git add .<br>
git commit -m "$@"<br>
git push -u pyjssp master<br><br>
{: .notice--info}

* github 참고사항
  * sh 실행
    * cd /Users/kimsanghun/MissFlash
    * sh github_push.sh "Modify-Post."
  * 초기 설정
    * Project 디렉토리로 이동
    * git config --global user.name "missflash"
    * git config --global user.email "missflash@gmail.com"
    * git config --global credential.helper store (로그인 정보는 ~/.git-credentials 에 저장)
  * 설정 확인
    * git config --list
  * 사용자 계정 삭제
    * Mac > 키체인 접근 > 모든 항목 > github.com 삭제
  * [Not Works!] 사용자 계정 삭제
    * --git credential-osxkeychain erase
  * [Not Works!] 사용자 비밀번호 변경
    * --git config --global --unset user.password
  * Git 저장소 생성 (.git 디렉토리 생성)
    * git init
  * 로컬 경로 이동
    * cd /Users/kimsanghun/MissFlash/Github
  * 리모트 저장소 연결
    * git remote add origin https://github.com/missflash/missflash.github.io.git
  * 리모트 저장소 확인
    * git remote -v
  * 다운로드
    * git pull origin master
    * git pull https://github.com/missflash/missflash.github.io.git master
    * // -- fatal: refusing to merge unrelated histories 에러 발생시 아래 명령 수행
    * // git pull origin master --allow-unrelated-histories
  * 로컬 경로 모든 파일 업로드 (\_post 폴더에 글 작성 후)
    * git add .
    * git commit -m "."
  * 업로드
    * git push -u origin master
    * git push https://github.com/missflash/missflash.github.io.git master
  * remote 제거
    * git remote remove origin
  * Commit 삭제 : [https://gmlwjd9405.github.io/2018/05/25/git-add-cancle.html](https://gmlwjd9405.github.io/2018/05/25/git-add-cancle.html)
    * git reset --hard HEAD~3
  * 100MB 이상 파일 Commit 불가
  * Your branch is ahead of 'origin/master' by 3 commits 에러 메시지 발생시
    * git push -u origin master 으로 정상 push 시도
    * 위 방법으로 안될 경우,
      * local 수정 파일을 다른 경로로 이동
      * git reset --hard origin/master 으로 remote의 버전으로 local 리셋
      * local 수정 파일 다시 복원
      * git push -u origin master 으로 정상 push 시도
