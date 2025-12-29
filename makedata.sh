#!/bin/bash
#
# Script to generate synthetic hologram dataset with many objects
#
./makegds.py --objects_dir objects_diatoms/small --data_dir data/lots --nimages 5000 --nobjects 400 --object_size_min 15 --object_size_range 50 --object_size_gamma 12 --composite_size 256 --composite_padding 0

./make_holograms_enhanced.py --data_dir data/lots --lambda_um .650 --z_mm 1 --dx_um 1.2 --model fresnel 
