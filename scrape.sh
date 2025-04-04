#!/bin/bash
url_125="https://webcams.nyctmc.org/api/cameras/566bce47-4390-4ff3-94ab-7a0ca2989163/image"
url_110="https://webcams.nyctmc.org/api/cameras/c217a64e-95c3-4442-96d1-7a6c177615d7/image"

while :
do
   t=$(date +%s)
   echo $t
   curl -s $url_110 > data/110/$t.jpg
   sleep 2
done
