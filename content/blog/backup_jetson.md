---
title: "Full backup Nvidia Jetson Xavier"
date: 2019-09-15T09:40:18+03:00
draft: false
---

## Command to do a full backup of the NVIDIA jetson xavier


[source is from here](https://devtalk.nvidia.com/default/topic/1039548/jetson-agx-xavier/xavier-cloning/3) 


### Creating the backup image file

You will need a PC running ubuntu with Nvidia sdk JetPack 4.2 and 64go of free space.
From this PC, ssh to your jetson (jetson is running normaly, not on restore mode).



```
ssh jetsonUser@JetsonIP
```


On the Jetson, through this ssh, stop the filesystem and force it to read only access:

```
echo u > /proc/sysrq-trigger
```

Still on the jetson, through ssh, transferring an image of full internal memory hard drive over ssh to host PC:

```
dd if=/dev/mmcblk0p1 | ssh user@hostpc dd of=/media/aya/usbaya/image.raw

```
_I am creating the raw img file in my external drive for space issue. I_


This will create on your PC a 30Gb file containing a full image of the Jetson. you can now turn of jetson.

Now on you pc you can convert the .raw image to a .img image file:

```
cd /home/aya/nvidia/nvidia_sdk/JetPack_4.2_Linux_P2888/Linux_for_Tegra/bootloader/

sudo ./mksparse -v --fillpattern=0 /media/aya/usbaya/image.raw /media/aya/usbaya/system.img
```
_I am creating the img file in my external drive for space issue. I_



### Restoring the image file

Boot the Jetson on restore mode (power and middle button for 5 sec), connect it with usb to the PC. 

```
cd /home/aya/nvidia/nvidia_sdk/JetPack_4.2_Linux_P2888/Linux_for_Tegra/bootloader/
```

copy in this bootloader folder your backup img file and rename it system.img 

```
sudo ./flash.sh -r jetson-xavier mmcblk0p1
```

done.