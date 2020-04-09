#!/bin/sh

wget -O /usr/sbin/gdrivedl 'https://f.mjh.nz/gdrivedl'
chmod +x /usr/sbin/gdrivedl
gdrivedl https://drive.google.com/open?id=1KyjhkLEw0D8zP8IiJTTOR0j6PGecKbqS /yolact/weights/res101_coco_800000.pth
# gdrivedl https://drive.google.com/open?id=1kMm0tBZh8NuXBLmXKzVhOKR98Hpd81ja /yolcat/weights/res50_coco_800000.pth
# gdrivedl https://drive.google.com/open?id=1Uwz7BYHEmPuMCRQDW2wD00Jbeb-jxWng /yolcat/weights/resnet50-19c8e357.pth
# gdrivedl https://drive.google.com/open?id=1vaDqYNB__jTB7_p9G6QTMvoMDlGkHzhP /yolcat/weights/resnet101_reducedfc.pth
