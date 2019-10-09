---
title: "Cities Skylines demo instruction "
date: 2019-10-09T13:25:18+03:00
draft: true
---

# Demo instruction 

## Smart city decentralized platform controlling traffic light

_Poster : "GIT\cities-skylines\poster.pptx"_


This demo shows riaps node controlling traffic light in the game Cities Skylines and sensing density around intersection.

You will need :
* 4 beaglebone black
* 1 router
* 1 raspberry pi
* Cities Skyline game
* The mods traffic manager president edition, modified, for the game cities skyline
* 1 or more LCD screen 16x2


When you ssh to bbb, if it doesn't ask for riaps login/passowrd, that mean it didn't boot from sd card. Un-plug power , then while you are pressing the sd boot button on bbb, plug it back. release button after 5 seconds. ssh login/pwd are riaps/riaps.

On all bbb running the default riaps image, install the library to control the lcd screen:
`sudo python3 -m pip install Adafruit_CharLCD`


![bbb_lcd_16x2](/img/cities/BBB_LCD.jpg)
![bbb_lcd_16x2_photo](/img/cities/IMG_2641.jpg)














































































On riaps VM, clone the repo :

```
cd git 
git clone https://github.com/kaust-merge/cities-skylines.git
```

clean the process running on the VM:


```
sudo pkill -SIGKILL riaps_device; sudo pkill -SIGKILL riaps_devm; sudo pkill -SIGKILL riaps_disco;  sudo pkill -SIGKILL riaps_deplo;sudo pkill -SIGKILL riaps_actor;  pkill -9 xterm; pkill -9 riaps_ctrl; pkill -9 rpyc_registry.py; pkill -9 redis-server; pkill -9 riaps_disco; pkill -9 riaps_devm;
```

launch riaps_ctrl in a terminal: 
```
riaps_ctrl
```

Open a terminal and create one instance of riaps_deplo :
```
sudo -E riaps_deplo
```

ssh to all 4 beaglebone and raspberry pi, open one terminal tab for each: 

```
ssh riaps@192.168.1.110
```

make sure all iot device are running only one instance of riaps_actor:

```
riaps@bbb-9c4a:~$ ps -e | grep riaps
 6435 ?        00:11:16 riaps_deplo
 6467 ?        00:00:09 riaps_disco
```

if not, kill them all:
```
sudo kill -9 6343 6348 6413 6414
```

display the log on all devices :
```
sudo journalctl -u riaps-deplo.service -f
```

On riaps_ctrl, select the folder path : `/home/riaps/git/cities-skylines/cities_riaps` , the .riaps and the .depl

You should all your 6 node now (4 bbb + 1 rpi + 1 on VM): 


![riaps_ctrl](/img/cities/mstsc_xVm4fh23c3.png)


On PC, launch the game cities skyline . The mod (in repository) should be copied in `C:\Users\AYACHIMI\AppData\Local\Colossal Order\Cities_Skylines\Addons\Mods`

Copy also the savageme demo.crp here 
`C:\Users\ayachimi\AppData\Local\Colossal Order\Cities_Skylines\Saves`

Load the savegame "demo" , go to north west, and add the 4 manual traffic light startight from north west and goind clockwise 

![riaps_ctrl](/img/cities/mstsc_9i1DdWZrHX.jpg)


On Vm, in riaps_ctrl, click on Deploy, then on Launch. Apps should be deployed and node become green:
![riaps_ctrl](/img/cities/mstsc_cypbK1n8MY.png)



Be sure that on each iot terminal output, there is no error :
beaglebone
![riaps_ctrl](/img/cities/mstsc_Qu1GBZn0da.png)
backup
![riaps_ctrl](/img/cities/mstsc_0wDI1tF1Ew.png)
rpi logger
![riaps_ctrl](/img/cities/mstsc_TVkMcR7mKM.png)
























































On PC , open the webapp grafana on raspberry pi IP address :
`http://192.168.1.192:3000/d/IEaNFaigk/cities?orgId=1&refresh=1s`
open "cities" dashboard (saved in repository/grafana.json) and set refresh rate at 1 sec with datas on last 5 min.
![riaps_ctrl](/img/cities/mstsc_F4KU6IP7bw.png)


You should see light state and density for each intersection.






















Now that everything is running, unplug ethernet cable of bbb actor 1 and 2.
After 3.5 sec, actor 1 backup (on VM) should take relay and data will be displayed again in grafana dashboard. Actor2 stay offline as it doesn't have a backup. In game you will soon see traffic jam on intersection south east.

All node connected and running: density is stable and stay below 25 :
![riaps_ctrl](/img/cities/chrome_eOtvVLkHtD.png)


No traffic jam in the game 
![riaps_ctrl](/img/cities/Cities_jfdYJfm7cC.jpg)




































Unplug ethernet cable of beaglebone actor 1 and actor 2 , see the 3.5 sec no-data for actor 1, then backup goes online and send datas. For Actor2 no backup and no datas.
![riaps_ctrl](/img/cities/chrome_krc1mj6pL7.png)






















Intersection managed by actor 2 start to have traffic jam, color becomming red:
![riaps_ctrl](/img/cities/Cities_14VKlxrTyn.jpg)




















Cars waiting at intersection : traffic light frozen to red. 
![riaps_ctrl](/img/cities/7PfQoZhBfR.jpg)




















Plug back ethernet cable, see the congestion digested by actor 2 , density going back to normal. 

![riaps_ctrl](/img/cities/chrome_k1XyFWs9kp.png)


Stop riaps app with riaps_ctrl, click on Stop then Remove. Check on iot device ssh output that everything has stopped. 