#!/bin/bash

# Run the hologram preparations and reconstruction

if [ "$VIRTUAL_ENV_PROMPT" != "holosurf" ]; then
	source .venv/bin/activate
fi
if [ "$VIRTUAL_ENV_PROMPT" != "holosurf" ]; then
	echo "Please activate the holosurf virtual environment first:"
	echo "  source ~/holosurf/venv/bin/activate"
	exit 1
fi

our_dialog=dialog
dataset=this

options=""
c=1
if [ -f data/.lastdir.txt ]; then
	def="--default-item `cat data/.lastdir.txt`"
else
	def=""
fi
# $our_dialog $def --menu  "Choose a dataset to process" 19 30 8 $options 2> /tmp/choice$$.txt
#if [[ $? != 0 ]]; then
#	echo "Cancelled"
#	exit 1
#fi
#dataset=$(cat /tmp/choice$$.txt)
#echo $dataset > data/.lastdir.txt

./src/holo-05-stream.py 
		$our_dialog --gauge "Preparing holograms..." 19 70 10

echo -e '\e[999;1H\n' # Move cursor to bottom
#
