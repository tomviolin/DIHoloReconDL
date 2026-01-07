#!/bin/bash

addr=10.24.31.245
timestamp=`date +%Y-%m%d-%H%M%S`
datahome=data/cap$timestamp
mkdir -p $datahome/raw
ffmpeg -i http://holoscope2.sfs.uwm.edu:81/stream -frames 1000 $datahome/raw/frame%05d.png -y

