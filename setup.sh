#!/usr/bin/env bash

our_dialog=dialog
# This script sets up a Python virtual environment and installs required packages.

# prompt for CUDA version
$our_dialog --title "dihm_unet Install" --menu "\n               Choose CUDA version for torch and CuPy\n " 15 70 8 CUDA-11.8 1 CUDA-12.1 2  2> /tmp/cuda_ver$$.txt
if [ $? -ne 0 ]; then
	echo -e "\e[999;1H\nCancelled."
	exit 1
fi
echo -e "\e[999;1H\n"
choice=$(cat /tmp/cuda_ver$$.txt)
if [ "$choice" == "CUDA-11.8" ]; then
	TORCH_CUDA_VER="-cu118"
	CUPY_CUDA_VER="cupy-cuda11x"
elif [ "$choice" == "CUDA-12.1" ]; then
	TORCH_CUDA_VER="-cu121"
	CUPY_CUDA_VER="cupy-cuda12x"
else
	echo -e "\e[999;1H\nBad selection."
	exit
fi

rm -f /tmp/cudaver$$.txt

# must get uv from astral.sh first because of uv's unique .venv structure
curl -LsSf https://astral.sh/uv/install.sh | sh
# --seed option populates venv with pip from python version
#        it does not install itself into the venv, unfortunately
# -p     specifies python version
uv venv --seed -p python3.13

source ./.venv/bin/activate
# ironically uv does not place itself in the venv, even with
# the --seed option, so we have to use plain pip to install uv. 
# Strange that the --seed option doesn't do that for us.
pip install uv
# now we can use uv to install the rest of the requirements
# torch
echo -e "\e[1m>>>\e[0m uv pip install -r requirements-torch$TORCH_CUDA_VER.txt"
uv pip install -r requirements-torch$TORCH_CUDA_VER.txt
# cupy
echo -e "\e[1m>>>\e[0m uv pip install -r requirements-$CUPY_CUDA_VER.txt"
uv pip install -r requirements-$CUPY_CUDA_VER.txt
# other requirements
uv pip install -r requirements.txt
# clone PyHoloscope where it needs to be for our code to find it
git clone https://github.com/MikeHughesKent/PyHoloscope.git

echo -e "\e[999;1H\nSetup complete. To activate the virtual environment, run:\n\n    source ./.venv/bin/activate\n"
echo "Place data files into data/ directory as needed."
echo "To run the system, use the command ./run.sh"

