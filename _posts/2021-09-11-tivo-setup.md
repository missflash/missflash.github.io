---
title: "TiVo Setup"
date: 2021. 9. 11. 오후 9:10:01
categories:
use_math: true
toc: true
toc_label: "Table of Contents"
toc_icon: "cog"
toc_sticky: true
---

[comment]: <> (포스트 화면 넓게 설정하고 싶을 때 추가, classes: wide)

* TiVo 최적화 방법
  * Ad Hoc 클라이언트 adbLink 설치 : [http://jocala.com/](http://jocala.com/)
    * [Windows](http://jocala.com/downloads/adblw43.exe)
    * [Mac](http://jocala.com/downloads/adblm43.dmg)
    * [Linux](http://jocala.com/downloads/adbll43.zip)
  * TiVo 개발자 환경 허용
    * 설정 > 기기 환경설정 > 정보 > 빌드 7번 선택
    * 설정 > 기기 환경설정 > 개발자 옵션 > USB 디버깅 활성화
    * 설정 > 네트워크 및 인터넷 > 연결된 네트워크에서 IP 주소 확인 (192.168.xxx.xxx)
  * adbLink 실행
    * Ad Hoc IP 입력 (192.168.xxx.xxx)
    * Connect 클릭
    * TiVo에서 USB 디버깅을 허용하시겠습니까 알람창 확인
      * 이 컴퓨터에서 항상 허용 체크
      * 확인 클릭
    * Connect > ADB Shell 클릭
    * Command 창에 아래 명령어 입력후 엔터
```
pm uninstall -k --user 0 com.utsmta.app
pm uninstall -k --user 0 com.tivo.tivoplusplayer
pm uninstall -k --user 0 com.tivo.tvlaunchercustomization
pm uninstall -k --user 0 com.droidlogic.overlay</li>
pm uninstall -k --user 0 com.nes.bugtracker
pm uninstall -k --user 0 com.nes.tvglobalkeyhandler
pm uninstall -k --user 0 com.nes.daemonservice
pm uninstall -k --user 0 com.nes.skywayclient
pm uninstall -k --user 0 com.droidlogic.SubTitleService
pm uninstall -k --user 0 com.limark.deviqcoreagent
pm disable-user --user 0 com.tivo.atom
```
