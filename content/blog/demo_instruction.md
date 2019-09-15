---
title: "Demo instruction "
date: 2019-09-08T17:23:18+03:00
draft: true
---

# Demo instruction 

## Demo 1 : Time synchronization 

_Poster : "Time synchroniztion with RIAPS"_

This demo shows the precision of RIAPS timesync service. 

You will need :
* 3 beaglebone black
* 1 router
* 1 oscilloscope

When you ssh to bbb, if it doesn't ask for riaps login/passowrd, that mean it didn't boot from sd card. Un-plug power , then while you are pressing the sd boot button on bbb, plug it back. release button after 5 seconds. ssh login/pwd are riaps/riaps.

On all bbb running the default riaps image, clone the riaps-timesync repository:

```
mkdir git
cd git
git clone 
git clone https://github.com/RIAPS/riaps-timesync.git
```

Set the bbb 1 to become master node for time synchronization:

```
timesyncctl config master
```

set bbb 2 and 3 to become slave:

```
timesyncctl config slave
```

on all bbb, check that everything is fine and launch the pulse generating app:

```
timesyncctl status
cd riaps-timesync/tools/sync_gen/
make
sudo ./sync_gen
```

the bbbs should start emitting one gpio pulse every second.

Start the oscilloscope and connect it to bbb2 and bbb3 on gpio P8_19 and ground. Load the setting pre-saved (aya3) in the oscilloscope, then with the button "Single", record the visible pulse on two signal. You will be able to zoom enough on graph to display the time delta between signal 1 and 2 : 252ns. You can do same with bbb1 and bbb2 to display time accuracy between master-slave clock.

## Demo 2: ML at the edge deployment

_Poster : "Fault-tolerance decentralized platform"_

This Demo is an example of machine learning deployed at the edge of the grid, running on IOT device, with fault tolerance, historian database recording, dashboard visualization, PMU C37.118 reading, running on riaps architecture.

You will need : 

* 2 beaglebone black running riaps default image
* 1 raspberry pi (model 3) running rasbian with riaps installed , grafana and influxDB
* 1 nvidia jetson xavier with riaps architecture installed and Tensorflow
* 1 PC running the riaps VM (password is : pokpokpok) and the PMU/PDC simulator
* 1 router

On all iot device , start by changing the riaps_disco service to use the python one (cpp version currently not sending message on jetson). You can SSH all devices from PC with putty.

```
sudo mv /usr/local/bin/riaps_disco /usr/local/bin/riaps_disco.bak
sudo mv /usr/local/bin/riaps_disco_redis /usr/local/bin/riaps_disco
sudo nano /etc/riaps.conf  ====> security off
```

On the nvidia jetson, launch :

```
sudo -E riaps_deplo
```

On the VM , copy the project "SASGdemo" , launch eclipse and from its menu launch riaps_ctrl.

![riaps_ctrl_icon](/instruction_screenshot/eclipse_launch_riaps_ctrl.png)

From riaps_ctrl, you should see the IP address of all the devices connected to the router. If a device does not appear, try to kill all riaps services:

```
ps -e | grep riaps
sudo pkill riaps
sudo kill -9 10121 10148 11094
```

![riaps_ctrl_ips](/instruction_screenshot/riaps_ctrl_ips.png)

to display logs on bbbs , launch :

```
sudo journalctl -u riaps-deplo.service -f
```

![deployment_iot](/instruction_screenshot/deployment_iot.png)

Deployment :

* 1 bbb as PMU reader (SensorLeaderNode) , extracting frequency, buffering it for 1 sec (30 values) and broadcasting it to network inside riaps message.
* 1 bbb as PMU reader backup (SensorBackupNode), wake up if no heartbeat from leader after 3.5 sec
* 1 raspberry pi as historian and dashboard (MonitorActor), receiving frequency and predictions from SensorLeader and Jetson .
* 1 Jetson Xavier (JetsonActor), receiving frequency , running Tensorflow to do prediction, and sending prediction result

Start the PMU/PDC simulator on PC:

![pdcsimulator](/instruction_screenshot/PDCSimulator.png)

On VM, in riaps_ctrl, select as folder : "/home/riaps/riaps-projects/SASGdemo" , as model : "SASGdemo.riaps" and as Depl : "SASGdemo.depl"

Click "Deploy" button, then launch SASGdemo :


![pdcsimulator](/instruction_screenshot/launchSASGdemo.png)

If everything goes right, you should see green background behind actors name :

![deployment](/instruction_screenshot/deploymentdone.png)

on bbb1 SSH output, you should see :

```
Sep 10 07:57:58 bbb-e59a RIAPS-DEPLO[5594]: [info]:[09-10-2019 07:57:58.347]:[23440]:SensorLeaderNode.leader:leader sending keepalive
Sep 10 07:57:59 bbb-e59a RIAPS-DEPLO[5594]: [info]:[09-10-2019 07:57:59.144]:[23440]:SensorLeaderNode.leader:leader : sending freq

```

on bbb2 you should see : 

```
Sep 10 07:58:14 bbb-9c4a RIAPS-DEPLO[11284]: [info]:[09-10-2019 07:58:14.384]:[28064]:SensorBackupNode.backup:backup: Keepalive message from leader received
Sep 10 07:58:15 bbb-9c4a RIAPS-DEPLO[11284]: [info]:[09-10-2019 07:58:15.378]:[28064]:SensorBackupNode.backup:backup: Keepalive message from leader received
```

on Jetson you should see:

```
[info]:[09-10-2019 10:58:50.301]:[20423]:JetsonActor.monitor: jetson received message
[info]:[09-10-2019 10:58:50.331]:[20423]:JetsonActor.monitor:PREDICTION ============> [1]
```

on raspberry pi you should see:

```
Sep 10 07:59:08 bbb-9078 RIAPS-DEPLO[14321]: [info]:[09-10-2019 07:59:08.617]:[2620]:MonitorActor.monitor: monitor received prediction
Sep 10 07:59:09 bbb-9078 RIAPS-DEPLO[14321]: [info]:[09-10-2019 07:59:09.601]:[2620]:MonitorActor.monitor: monitor received message
```

As raspberry pi is running Grafana dashboard, you access to it with any web browser : http://192.168.1.192:3000

Open MERGE dashboard , you should see the datas :

![deployment](/instruction_screenshot/grafana.png)

On PMU simulator, try to change frequency shape :


![deployment](/instruction_screenshot/freq_shape.png)

Jetson prediction should change and grafana dashboard will look like that :

![deployment](/instruction_screenshot/grafana2.png)

To test backup mechanism, try to unplug ethernet port of bbb1 , bbb2 should become leader :

bbb2 output :
```
Sep 10 08:04:48 bbb-9c4a RIAPS-DEPLO[11284]: [info]:[09-10-2019 08:04:48.157]:[28064]:SensorBackupNode.backup:leader : sending freq
Sep 10 08:04:48 bbb-9c4a RIAPS-DEPLO[11284]: [info]:[09-10-2019 08:04:48.559]:[28064]:SensorBackupNode.backup:No keepalive messages received for the timeout, backup assuming leader role
Sep 10 08:04:49 bbb-9c4a RIAPS-DEPLO[11284]: [info]:[09-10-2019 08:04:49.175]:[28064]:SensorBackupNode.backup:leader : sending freq
Sep 10 08:04:50 bbb-9c4a RIAPS-DEPLO[11284]: [info]:[09-10-2019 08:04:50.201]:[28064]:SensorBackupNode.backup:leader : sending freq
Sep 10 08:04:51 bbb-9c4a RIAPS-DEPLO[11284]: [info]:[09-10-2019 08:04:51.231]:[28064]:SensorBackupNode.backup:leader : sending freq
Sep 10 08:04:52 bbb-9c4a RIAPS-DEPLO[11284]: [info]:[09-10-2019 08:04:52.060]:[28064]:SensorBackupNode.backup:No keepalive messages received for the timeout, backup assuming leader role
Sep 10 08:04:52 bbb-9c4a RIAPS-DEPLO[11284]: [info]:[09-10-2019 08:04:52.249]:[28064]:SensorBackupNode.backup:leader : sending freq
Sep 10 08:04:53 bbb-9c4a RIAPS-DEPLO[11284]: [info]:[09-10-2019 08:04:53.268]:[280
```

After few second, grafana dashboard will display frequency comming from bbb2 backup
![bbb1_unplug](/instruction_screenshot/bbb1_unpluged.png)




bbb1 output after reconnecting ethernet cable
```
Sep 10 08:06:03 bbb-e59a RIAPS-DEPLO[5594]: [info]:[09-10-2019 08:06:03.881]:[23440]:SensorLeaderNode.leader:leader sending keepalive
Sep 10 08:06:04 bbb-e59a RIAPS-DEPLO[5594]: [info]:[09-10-2019 08:06:04.046]:[23440]:SensorLeaderNode.leader:leader : sending freq

```

bbb2 output after bbb1 has been reconnected
```
Sep 10 08:06:19 bbb-9c4a RIAPS-DEPLO[11284]: [info]:[09-10-2019 08:06:19.906]:[28064]:SensorBackupNode.backup:backup: Keepalive message from leader received
Sep 10 08:06:20 bbb-9c4a RIAPS-DEPLO[11284]: [info]:[09-10-2019 08:06:20.905]:[28064]:SensorBackupNode.backup:backup: Keepalive message from leader received

```

After reconnecting bbb1 LeaderSensor, bbb2 should become backup 
![bbb1_replug](/instruction_screenshot/bbb1_repluged.png)

Stop the riaps and remove app from riaps_ctrl at the end of the demo.
















## Demo 3 : Security mechanism

_Poster : "Decentralized architecture vision"_

You will need :

* Demo 2 running
* 1 pc with Wireshark installed

Launch Demo 2 , when it's running, launch wireshark on PC and filter IP address to only show jetson message :

```
ip.src == 192.168.1.147
```

Click on a SSL packet :

![bbb1_replug](/instruction_screenshot/wireshark.png)

Everything should be encrypted

