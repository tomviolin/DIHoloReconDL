#!/usr/bin/env python3

def updateprog(percentage, msg=""):
    print (percentage,flush=True)
    print(f"XXX\n{msg}\nXXX",flush=True)

updateprog(0,"loading os,sys,logging")
import os,sys,logging
updateprog(10,"loading cv2")
import cv2
updateprog(20,"loading numpy")
import numpy as np
updateprog(30,"loading cupy")
import cupy
import cupy as cp
updateprog(40,"loading pyholoscope")
import context
import pyholoscope as pyh

updateprog(100,"loading done")


def global_exception_handler(exctype, value, tb):
    import traceback
    tb_lines = traceback.format_exception(exctype, value, tb)
    tb_text = ''.join(tb_lines)
    os.system('stty sane');
    print("\x1b[999;1H\nAn unhandled exception occurred:\n", tb_text, flush=True, file=sys.stderr)
    sys.exit(1)

sys.excepthook = global_exception_handler

def force_exit(msg):
    os.system('stty sane');
    print(msg, flush=True, file=sys.stderr)
    sys.exit(1)


imgsum = None
imgavg = None

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
    datahst = cp.histogram(data, bins=256, range=(0.0, 1.0))[0]
    # find most common value
    mostcommon = np.argmax(datahst)/256
    # shift so that most common value is at 0.5

    n = np.log(0.5) / np.log(mostcommon)
    data = data ** n

    return data

fixhl = fixhololevel


# holo params
wavelen = 650.0e-9
dx = 1.12e-6



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




#img_size = 512

#r = cp.sqrt(x*x + y*y) * dx

zees = np.arange(wavelen*50, 0.01, wavelen*1) 

def makefig(paddedholo, zi):
    # these will change
    global zees, r
    zi=np.clip(zi, 0, len(zees)-1)
    z = zees[zi]
    #d = cp.sqrt(r*r+z*z)

    holo.set_depth(z)
    # Refocus
    
    if type(paddedholo) is np.ndarray:
        paddedholo = cp.array(paddedholo)
    if len(paddedholo.shape) > 2:
        paddedholo = paddedholo[...,1]
    paddedholo = paddedholo.astype(cp.float32)
    paddedholo = fixhololevel(paddedholo)

    paddedholo = fixhl((paddedholo + 0.01) / (imgavg + 0.01))

    cimage = holo.process(paddedholo)

    return cimage

datahome,_ = os.path.split(sys.argv[1])
datahome,_ = os.path.split(datahome)
### main program logic here ###
imgptr = 1
zi = 0
updateprog(0,"Checking for curpos.csv")
if os.path.exists(f"{datahome}/curpos.csv"):
    pos = open(f"{datahome}/curpos.csv","r").read().strip().split(',')
    imgptr = int(pos[0])
    zi = int(pos[1])



for imgp in range(1, len(sys.argv)):
    if not os.path.exists(sys.argv[imgp]):
        updateprog(f"\x1b[36;1mFile not found: {sys.argv[imgp]}\x1b[0m")
        continue
    percentage = int(100 * (imgp - 1) / (len(sys.argv) - 2))
    updateprog(percentage,f"Loading images: {sys.argv[imgp]}")
    hologram = cv2.imread(sys.argv[imgp])
    hologram = fixhololevel(hologram)
    if imgsum is None:
        imgsum = hologram
    else:
        imgsum += hologram

imgavg = fixhololevel(imgsum)
cimgavg = cp.array(imgavg)
updateprog(100,"Images loaded, average computed")
print("100\nXXX",flush=True)    
escaped = False
windowCreated = False
windowName = "image"
showEdges = False
while not escaped:
    if len(sys.argv) < 2:
        break
    ### updateprog(0,f"Loading image: {sys.argv[imgptr]}")
    hologram = cp.array(cv2.imread(sys.argv[imgptr]))

    #hologram = fixhololevel(hologram)
    datahome, basename = os.path.split(sys.argv[imgptr])
    datahome, datalastdir = os.path.split(datahome)

    paddedholo = hologram.copy()

    cimage = makefig(paddedholo, zi)

    hist = np.histogram(np.abs(cimage), bins=256, range=(0.0, 1.0))[0]

    if type(cimage) is np.ndarray:
        cimag2 = np.abs(cimage.copy())
    else:
        cimag2 = np.abs(cimage.get())


    # focus measure using difference of gaussian edges


    #updateprog(33,f"processing image: {sys.argv[imgptr]}")
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
    #updateprog(66,f"annotating image: {sys.argv[imgptr]}")
    #cimag2[...,2] = edges.copy()
    ##cimag2[...,1] = edges.copy()
    #cimag2[...,0] = edges.copy()
    cv2.putText(cimag2,f"z={zees[zi]:09.06f} zi={zi:04d} frame={os.path.basename(sys.argv[imgptr])}", (1,51),cv2.FONT_HERSHEY_DUPLEX,0.8,(0,0,0),3, cv2.LINE_AA)
    cv2.putText(cimag2,f"z={zees[zi]:09.06f} zi={zi:04d} frame={os.path.basename(sys.argv[imgptr])}", (1,51),cv2.FONT_HERSHEY_DUPLEX,0.8,(55,255,255),1, cv2.LINE_AA)
    bins = len(hist)
    for i in range(bins):
        xcoord = int(i * cimag2.shape[1] / bins)
        ycoord = int(cimag2.shape[0] - int(np.log(1+(hist[i]))) * (cimag2.shape[0]/8) / max(np.log(1+hist)))
        cv2.rectangle(cimag2, ( xcoord, cimag2.shape[0]),( xcoord+int(1/bins*cimag2.shape[1]), ycoord), (0,255,255), -1)
    if type(cimag2) is not np.ndarray:
        cimag2 = cimag2.get()
    #updateprog(100,f"displaying image: {sys.argv[imgptr]}")
    while True:
        if playing:
            if imgptr < len(sys.argv)-1:
                imgptr += 1
            else:
                imgptr = 1
        if not windowCreated:
            cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(windowName, 800,800)
            cv2.setWindowTitle(windowName, "Hologram Viewer - Press 'q' or ESC to quit")
            cv2.imshow(windowName, cimag2)
            cv2.moveWindow(windowName, 0,0)
            windowCreated = True
        else:
            cv2.imshow(windowName, cimag2)
        k=cv2.waitKey(33)
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
            if imgptr < len(sys.argv)-1:
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
