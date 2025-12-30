# -*- coding: utf-8 -*-
"""
More detailed example of how to use inline holography functionality of
PyHoloscope.

See inline_example.py for a more minimal example.

This example loads an inline hologram and a background image (i.e. with the
sample removed).

The images are loaded using the PyHoloscope 'load_image' function.
Alternatively you can load these in using any method that results in them
being stored in a 2D numpy array.

We instantiate a 'Holo' object and then pass in the system parameters and
various options.

We call the 'process' method of 'Holo' to refocus the hologram. If you have
a GPU and CuPy is installed the GPU will be used, otherwise it will revert to
CPU.

We then add normalisation, background subtraction and windowing.

Finally we use the 'amplitude' function to extract the amplitude of the
refocused image for display.

"""

from time import perf_counter as timer
from matplotlib import pyplot as plt

import context  # Loads relative paths

import pyholoscope as pyh

from pathlib import Path
import os,sys

wavelen = 635.0e-9
dx = 1.12e-6

if len(sys.argv) < 2:
    print(f"usage: {sys.argv[0]} datadir")
    sys.exit(1)

datahome = sys.argv[1]



# Load hologram and background images
holoFile = Path("../test/integration_tests/test data/inline_example_holo.tif")
backFile = Path("../test/integration_tests/test data/inline_example_back.tif")

hologram = pyh.load_image(holoFile)
background = pyh.load_image(backFile)


# Create an instance of the Holo class
holo = pyh.Holo(
    mode=pyh.INLINE,  # For inline holography
    wavelength=wavelen,  # Light wavelength, m
    pixel_size=dx,  # Hologram physical pixel size, m
#    background=background,  # To subtract the background
    depth=0.0130,
)  # Distance to refocus, m

# Refocus
recon = holo.process(hologram)



""" With normalisation"""
# We now add normalisation, we could have done this when we created the
# Holo object, by passing in normlise = backHologram, but we can also add this
# in later as follows:
holo.set_normalise(background)

# Refocus
reconNorm = holo.process(hologram)


""" With background and normalisation """
# We now add background subtraction, we could have done this is when we created
# the Holo object, by passing in background = backHologram, but we can also add
# this in later as follows:
holo.set_background(background)

# Refocus
reconNormBack = holo.process(hologram)


""" With background and normalisation and window """
# We now add a cosine window to reduce edge artefacts, we could have done this is when
# we created the Holo object, by passing in autoWindow = True, but we can also add this
# in later as follows:
holo.set_auto_window(True)

# By defualt the skin thickness (distance over which the window smoothly
# changes from transparent) to opaque) is 10 pixels, but we can set a different
# value
holo.set_window_thickness(20)

# We pre-compute the window, this is optional and would be done the next time we call
# process. We have to pass in either the background or the hologram so that holo
# knows how large to make the window.
holo.update_auto_window(background)

# Refocus
reconNormBackWindow = holo.process(hologram)


""" Refocusing to a different depth """
# We now refocus the hologram to a different depth. We change the refocus depth using:
holo.set_depth(0.01)

# We could call update_propagator() here, but we don't have to as PyHoloscope will
# realise the depth has changed and regenerate the propagator when we called process.
# This process will take a little longer the first time we call it since we are
# generation the propagator.

reconNormBackWindow2 = holo.process(hologram)


""" Display results """
plt.figure(dpi=150)
plt.title("Raw Hologram")
plt.imshow(hologram, cmap="gray")
plt.savefig('inadv_raw.png')
plt.close('all')

plt.figure(dpi=150)
plt.title("Refocused Hologram")
plt.imshow(pyh.amplitude(recon), cmap="gray")
plt.savefig('inadv_refocused.png')
plt.close('all')

plt.figure(dpi=150)
plt.title("Refocused Hologram with \n Normalisation")
plt.imshow(pyh.amplitude(reconNorm), cmap="gray")
plt.savefig('inadv_refocused_w_normalization.png')
plt.close('all')

plt.figure(dpi=150)
plt.title("Refocused Hologram with Background \n and Normalisation")
plt.imshow(pyh.amplitude(reconNormBack), cmap="gray")
plt.savefig('inadv_refocused_w_background_and_normalization.png')
plt.close('all')

plt.figure(dpi=150)
plt.title("Refocused Hologram with Background, \nNormalisation and Windowing")
plt.imshow(pyh.amplitude(reconNormBackWindow), cmap="gray")
plt.savefig('inadv_refocused_w_background_and_normalization_and_windowing.png')
plt.close('all')

plt.figure(dpi=150)
plt.title(
    "Refocused Hologram (Wrong Depth) with Background, \nNormalisation and Windowing"
)
plt.imshow(pyh.amplitude(reconNormBackWindow2), cmap="gray")
plt.savefig('inadv_refocused_wrong_depth_w_background_and_normalization_and_windowing.png')
plt.close('all')

