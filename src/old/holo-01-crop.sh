#!/bin/bash

datahome=data/$1
echo $datahome
mkdir -p $datahome/cropped
for f in $datahome/raw/*.png; do
	convert $f -thumbnail 512 -crop 512x512+0+128 $datahome/cropped/`basename $f`
	echo $f cropped.
done 

