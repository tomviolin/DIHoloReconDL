#!/usr/bin/env python3

import sys,os,shutil
from glob import glob
from dialog import Dialog
import locale
locale.setlocale(locale.LC_ALL, '')
dlg = Dialog(dialog="dialog")
# sset up dialog to have no background
dlg.set_background_title("DIHM UNet Trainer")

dlg.gauge_start("Loading modules...", width=70, height=19,percent=10)

def updateprog(percentage, msg=""):
    dlg.gauge_update(percentage, msg)

updateprog(0,"loading os,sys,logging")
import os,sys,logging
updateprog(10,"loading cv2")
import cv2
updateprog(20,"loading numpy")
import numpy as np
updateprog(40,"loading cupy")
import cupy

import cupy as cp
updateprog(60,"loading pyholoscope")
import context
import pyholoscope as pyh

updateprog(100,"loading done")

dlg.gauge_stop()

def global_exception_handler(exctype, value, tb):
    import traceback
    tb_lines = traceback.format_exception(exctype, value, tb)
    tb_text = ''.join(tb_lines)
    os.system('stty sane');
    print("\x1b[999;1H\nA bad thing happened:\n", tb_text, flush=True, file=sys.stderr)
    sys.exit(1)


sys.excepthook = global_exception_handler


def force_exit(msg):
    os.system('reset')
    os.system('stty sane');
    print(msg, flush=True, file=sys.stderr)
    sys.exit(1)


imgsum = None
imgavg = None
imgarray = []
cimag2k = None
playing = True

def fixhololevel(pdata):
    data = None
    # grayscale only
    if len(pdata.shape) > 2:
        data = cp.array((pdata[...,1]).astype(cp.float32))
        # data = cp.float32(cv2.cvtColor(pdata,cv2.COLOR_BGR2GRAY))
    else:
        data=pdata.copy()
    if type(data) is np.ndarray:
        data = cp.array(data)
    data = cp.abs(data) 
    # normalize to 0..1
    data -= data.min()
    data /= data.max()
    # compute histogram
    datahst = cp.histogram(data.flatten(), bins=256, range=(0.0, 1.0))[0]
    # find most common value
    mostcommon = np.argmax(datahst)/256
    # shift so that most common value is at 0.5

    n = np.log(0.5) / np.log(np.median(data.get()))
    data = data ** n

    return data

fixhl = fixhololevel


# holo params
wavelen = 650.0e-9
dx = 1.12e-6

rawdatasets = sorted(glob("data/*/holo/"))
datasets = []
for d in rawdatasets:
    dh = d.split('/')[1]
    if not os.path.exists(f"data/{dh}/holo"):
        continue
    count = len(glob(f"data/{dh}/holo/*.png")) + len(glob(f"data/{dh}/holo/*.jpg")) + len(glob(f"data/{dh}/holo/*.jpeg"))
    if count == 0:
        continue
    desc = f"[{count} images]"
    if os.path.exists(f"data/{dh}/readme.txt"):
        desc = desc + " " + open(f"data/{dh}/readme.txt","r").read().strip()
    datasets.append((dh, desc))
print(datasets)
ok,choice = dlg.menu("Choose dataset", choices=[d for d in datasets], width=78)
if not ok:
    force_exit("No dataset selected, exiting.")

datahome = f"data/{choice}"

imgarray = glob(f"{datahome}/holo/*.png") + glob(f"{datahome}/holo/*.jpg") + glob(f"{datahome}/holo/*.jpeg")
imgarray = sorted(imgarray)
if len(imgarray) == 0:
    force_exit("No images found in dataset, exiting.")


#img_size = 512

#r = cp.sqrt(x*x + y*y) * dx

zees = np.arange(wavelen*0, 0.01, wavelen*1) 

def makefig(paddedholo, zi):
    # these will change
    global zees, r, imgavg, holo
    zi=np.clip(zi, 0, len(zees)-1)
    z = zees[zi]
    #d = cp.sqrt(r*r+z*z)

    holo.set_depth(z)
    # Refocus
    """
    if type(paddedholo) is np.ndarray:
        paddedholo = cp.array(paddedholo)
    if len(paddedholo.shape) > 2:
        paddedholo = paddedholo[...,1]
    paddedholo = paddedholo.astype(cp.float32)
    """
    paddedholo = fixhololevel(paddedholo)
    """
    ia = (imgavg - 0.5) * 0.95  + 0.5
    ph = (paddedholo - 0.5) * 0.95+ 0.5
    paddedholo = fixhl(ph / ia)
#    paddedholo = fixhl((paddedholo + 0.1) / (imgavg + 0.1))
    """
    cimage = holo.process(paddedholo)

    return cimage

### main program logic here ###
imgptr = 1
zi = 0
if os.path.exists(f"{datahome}/curpos.csv"):
    pos = open(f"{datahome}/curpos.csv","r").read().strip().split(',')
    imgptr = int(pos[0])
    zi = int(pos[1])



dlg.gauge_start(f"Loading images...", width=70, height=19,percent=0)
for imgp in range(0, len(imgarray)):
    if not os.path.exists(imgarray[imgp]):
        updateprog(f"\x1b[36;1mFile not found: {imgarray[imgp]}\x1b[0m")
        continue
    percentage = int(100 * (imgp - 1) / (len(imgarray) - 2))
    updateprog(percentage,f"Loading images: {imgarray[imgp]}")
    hologram = cp.array(cv2.imread(imgarray[imgp], cv2.IMREAD_GRAYSCALE), dtype=cp.float32)
    hologram = fixhololevel(hologram)
    if imgsum is None:
        imgsum = hologram
    else:
        imgsum += hologram
dlg.gauge_stop()
imgavg = fixhololevel(imgsum)
cimgavg = cp.array(imgavg)
cv2.imshow("Average Hologram", (imgavg.get() * 255).astype(np.uint8))
cv2.imwrite("frameavg.png", (imgavg.get() * 255).astype(np.uint8))


# Create an instance of the Holo class
holo = pyh.Holo(
    mode=pyh.INLINE,  # For inline holography
    wavelength=wavelen,  # Light wavelength, m
    pixel_size=dx,  # Hologram physical pixel size, m
#    background=background,  # To subtract the background
    #depth=zees[zi],
    invert=False,
    cuda=True,
    #background=imgavg
)  # Distance to refocus, m




escaped = False
windowCreated = False
windowName = "image"
showEdges = False
while not escaped:
    if len(imgarray) < 2:
        break
    ### updateprog(0,f"Loading image: {imgarray[imgptr]}")
    hologram = cp.array(cv2.imread(imgarray[imgptr], cv2.IMREAD_GRAYSCALE), dtype=cp.float32)

    #hologram = fixhololevel(hologram)
    datahome, basename = os.path.split(imgarray[imgptr])
    datahome, datalastdir = os.path.split(datahome)

    paddedholo = hologram.copy()

    cimage = makefig(paddedholo, zi)

    hist = np.histogram(np.abs(cimage), bins=256, range=(0.0, 1.0))[0]

    if type(cimage) is np.ndarray:
        cimag2 = np.abs(cimage.copy())
    else:
        cimag2 = np.abs(cimage.get())


    # focus measure using difference of gaussian edges


    #updateprog(33,f"processing image: {imgarray[imgptr]}")
    edges = cv2.GaussianBlur(cimag2, (5,5),4)
    edges2= cv2.GaussianBlur(cimag2, (5,5),8)
    edges = cv2.absdiff(edges, edges2)
    edges = cv2.normalize(edges, None, 0, 255, cv2.NORM_MINMAX)

    zi = np.clip(zi, 0, len(zees)-1)
    cimag2 = fixhololevel(cimag2).get()
    
    peaks  = cimag2 < np.quantile(cimag2,0.004)
    cimag2 = cv2.cvtColor((cimag2*255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    #cimag2[...,1][peaks] = 255
    if showEdges: cimag2[...,1]=edges
    #cimag2 = cv2.medianBlur(cimag2, 5)
    #updateprog(66,f"annotating image: {imgarray[imgptr]}")
    #cimag2[...,2] = edges.copy()
    ##cimag2[...,1] = edges.copy()
    #cimag2[...,0] = edges.copy()
    cv2.putText(cimag2,f"z={zees[zi]:09.06f} zi={zi:04d} frame={os.path.basename(imgarray[imgptr])}", (1,51),cv2.FONT_HERSHEY_DUPLEX,0.8,(0,0,0),3, cv2.LINE_AA)
    cv2.putText(cimag2,f"z={zees[zi]:09.06f} zi={zi:04d} frame={os.path.basename(imgarray[imgptr])}", (1,51),cv2.FONT_HERSHEY_DUPLEX,0.8,(55,255,255),1, cv2.LINE_AA)
    bins = len(hist)
    for i in range(bins):
        xcoord = int(i * cimag2.shape[1] / bins)
        ycoord = int(cimag2.shape[0] - int(np.log(1+(hist[i]))) * (cimag2.shape[0]/8) / max(np.log(1+hist)))
        cv2.rectangle(cimag2, ( xcoord, cimag2.shape[0]),( xcoord+int(1/bins*cimag2.shape[1]), ycoord), (0,255,255), -1)
    if type(cimag2) is not np.ndarray:
        cimag2 = cimag2.get()
    #updateprog(100,f"displaying image: {imgarray[imgptr]}")
    while True:
        if playing:
            if imgptr < len(imgarray)-1:
                imgptr += 1
            else:
                imgptr = 1
        """
        cimag2 = np.clip(cimag2, 127, 255)
        cimag2 = cimag2 - 127
        cimag2 = cimag2 * 2
        cimag2 = cimag2
        """
        if not windowCreated:
            cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(windowName, 800,800)
            cv2.setWindowTitle(windowName, "Hologram Viewer - Press 'q' or ESC to quit")
            cv2.imshow(windowName, cimag2)
            cv2.moveWindow(windowName, 0,0)
            windowCreated = True
        else:
            cv2.imshow(windowName, cimag2)
        k=cv2.waitKey(13)
        if k == ord('q') or k == 27:
            escaped = True
            break



        print(f"{imgptr},{zi}", file=open(f"{datahome}/_curpos.csv", "w"))
        os.rename(f"{datahome}/_curpos.csv", f"{datahome}/curpos.csv")
        # navigation keys
        #
        #   keyboard layout:
        # y u i      <-- higher z (by 100,10,1)
        # h j k      <-- lower z  (by 100,10,1)
        #      . ,   (next/prev image)
        # , - previous image
        if k == ord(',') or k == 81 or k == 8:
            if imgptr > 1:
                imgptr -= 1
                break
        # . - next image
        if k == ord('.') or k == 83:
            if imgptr < len(imgarray)-1:
                imgptr += 1
                break

        # k - lower z value
        if k == ord('k'):
            if zi > 0:
                zi -= 1
                break
        # i - higher z value
        if k == ord('i'):
            if zi < len(zees) - 1:
                zi += 1
                break

        # j - lower by 10 z values
        if k == ord('j') or k == 84:
            if zi > 10:
                zi -= 10
            else:
                zi  = 0
            break
        # u - higher by 10 z values
        if k == ord('u') or k == 82:
            if zi < len(zees) - 11:
                zi += 10
            else:
                zi = len(zees)-1
            break

        # h - lower by 100 z values
        if k == ord('h'):
            if zi > 100:
                zi -= 100
            else:
                zi  = 0
            break
        # y - higher by 100 z values
        if k == ord('y'):
            if zi < len(zees) - 101:
                zi += 100
            else:
                zi = len(zees)-1
            break

        if k == ord('e'):
            showEdges = not showEdges;
            break

        if k == 32:
            playing = not playing
            break

        break
