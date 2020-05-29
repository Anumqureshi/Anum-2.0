#!/bin/bash

head -n542 XBTUSD-1d-0320-700.csv > interim_data1.csv
 
for ((i=542; i<702; i++))
do

    head -$i XBTUSD-1d-0320-700.csv > interim_data1.csv
    python "fbp.py"
done