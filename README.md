# DIHoloReconDL
Digital Inline Holographic Reconstruction with Deep Learning


## setup.sh

Initial setup of the DIHoloReconDL Python environment.


## makegds

#### usage: **`./makegds.py [ -h | --help ]`**

Generates ground-truth files for hologram simulation. bsically, it takes
files from objects_diatoms and scatters them randomly in a larger field of view.

## 

Usage: **`make_holograms_advanced.py [ -h | --help ]`**

Simulates inline holograms from ground-truth files using the angular spectrum method.

## dihm_unet.py [ -h | --help ]

NOTE: This script is fully *dialog* based, and is best run without command-line arguments.

This runs the training and testing of the U-Net model.  Takes the files generated
by makegds.py and make_holograms_advanced.py as input.  During training, the
system uses the holograms as input and the ground-truth files as targets or labels.

Also during training, the system periodically tests the perforamance of the model
on a separate set of holograms and ground-truth files that were set aside before training. This
is called "validation" and helps to monitor how well the model is learning.

