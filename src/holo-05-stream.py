#!/usr/bin/env python3

def updateprog(percentage, msg=""):
    print (percentage,flush=True)
    print(f"XXX\n{msg}\nXXX",flush=True)

updateprog(0,"loading os,sys,logging")
import os,sys,logging
updateprog(20,"loading cv2")
import cv2
updateprog(40,"loading numpy")
import numpy as np
updateprog(60,"loading cupy")
import cupy
import cupy as cp
updateprog(80,"loading pyholoscope")
import context
import pyholoscope as pyh

updateprog(100,"loading done")


def global_exception_handler(exctype, value, tb):
    import traceback
    tb_lines = traceback.format_exception(exctype, value, tb)
    tb_text = '\n'.join(tb_lines)
    os.system('stty sane');
    updateprog(99,f"A bad thing happened:\n{tb_text}")
    sys.exit(1)

sys.excepthook = global_exception_handler

def force_exit(msg):
    os.system('stty sane');
    print(msg, flush=True, file=sys.stderr)
    sys.exit(1)


imgsum = None
imgavg = None
finavg = None
reconavg = None

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


#img_size = 512

# holo params
wavelen = 650.0e-9
dx = 5.12e-6
#r = cp.sqrt(x*x + y*y) * dx

zees = np.arange(0.0000, 0.01, wavelen*2) 


# Create an instance of the Holo class
holo = pyh.Holo(
    mode=pyh.INLINE,  # For inline holography
    wavelength=wavelen,  # Light wavelength, m
    pixel_size=dx,  # Hologram physical pixel size, m
#    background=background,  # To subtract the background
#    depth=zees[zi],
    invert=False,
    cuda=True,
#   background=imgavg
)  # Distance to refocus, m






def makefig(holoin, zi):
    # these will change
    global zees, r, imgavg, finavg
    zi=np.clip(zi, 0, len(zees)-1)
    z = zees[zi]
    #d = cp.sqrt(r*r+z*z)


    if imgavg is None:
        imgavg = holoin.copy()
    else:
        alpha = 0.01
        imgavg = imgavg * (1.0-alpha) + holoin * alpha

    paddedholo = holoin.copy()
    
    # Refocus
    
    if type(paddedholo) is np.ndarray:
        paddedholo = cp.array(paddedholo)
    if len(paddedholo.shape) > 2:
        paddedholo = paddedholo[...,1]
    paddedholo = paddedholo.astype(cp.float32)
    paddedholo = fixhololevel(paddedholo)
    holo.set_depth(zees[zi])

    #holo.set_background(cp.array(imgavg))
    #paddedholo = fixhololevel((paddedholo + 0.01) / (imgavg + 0.01))
    cimage = holo.process(paddedholo)
    #cimage = holo.process(finavg)
    return cimage

#datahome,_ = os.path.split(sys.argv[1])
#datahome,_ = os.path.split(datahome)
### main program logic here ###
image_counter = 0
zi = 0
"""
updateprog(0,"Checking for curpos.csv")
if os.path.exists(f"{datahome}/curpos.csv"):
    pos = open(f"{datahome}/curpos.csv","r").read().strip().split(',')
    imgptr = int(pos[0])
    zi = int(pos[1])

"""


cap  = cv2.VideoCapture("http://esp32s3-f109e8.wifi.uwm.edu:81/stream", )


capok,capframe = cap.read()
"""
while ok:
    if not os.path.exists(sys.argv[imgp]):
        force_exit(f"File not found: {sys.argv[imgp]}")
    #percentage = int(100 * (imgp - 1) / (len(sys.argv) - 2))
    #updateprog(percentage,f"Loading images: {sys.argv[imgp]}")
    hologram = cv2.imread(sys.argv[imgp])
    hologram = fixhololevel(hologram)
    if imgsum is None:
        imgsum = hologram
    else:
        imgsum += hologram

imgavg = fixhololevel(imgsum)
#updateprog(100,"Images loaded, average computed")
print("100\nXXX",flush=True)
"""
escaped = False
windowCreated = False
windowName = "image"
showEdges = False


while capok and (not escaped):
    updateprog(50,f"Loading image: {image_counter}")
    capframe = cv2.cvtColor(capframe, cv2.COLOR_BGR2GRAY)
    hologram = cp.float32(capframe)/255.0

    #hologram = fixhololevel(hologram)
    #datahome, basename = os.path.split(sys.argv[imgptr])
    #datahome, datalastdir = os.path.split(datahome)

    paddedholo = hologram.copy()

    cimage = makefig(paddedholo, zi)

    hist = np.histogram(np.abs(cimage), bins=cimage.shape[1], range=(0.0, 1.0))[0]

    if type(cimage) is np.ndarray:
        cimadj = np.abs(cimage.copy())
    else:
        cimadj = np.abs(cimage.get())
    # focus measure using difference of gaussian edges

    updateprog(60,f"processing image: {image_counter}")
    #edges = cv2.GaussianBlur(cimadj, (5,5),4)
    #edges2= cv2.GaussianBlur(cimadj, (5,5),8)
    #edges = cv2.absdiff(edges, edges2)
    #edges = cv2.normalize(edges, None, 0, 255, cv2.NORM_MINMAX)

    zi = np.clip(zi, 0, len(zees)-1)
    cimadj = fixhololevel(cimadj).get()
    
    peaks  = cimadj < np.quantile(cimadj,0.004)
    cimadj = cv2.cvtColor((cimadj*255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    #cimadj[...,1][peaks] = 255
    #if showEdges: cimadj[...,1]=ges
    #cimadj = cv2.medianBlur(cimadj, 5)
    updateprog(88,f"annotating image: {image_counter}")
    #cimadj[...,2] = edges.copy()
    ##cimadj[...,1] = edges.copy()
    #cimadj[...,0] = edges.copy()
    cv2.putText(cimadj,f"z={zees[zi]:09.06f} zi={zi:04d} frame={image_counter}", (1,51),cv2.FONT_HERSHEY_DUPLEX,0.8,(0,0,0),3, cv2.LINE_AA)
    cv2.putText(cimadj,f"z={zees[zi]:09.06f} zi={zi:04d} frame={image_counter}", (1,51),cv2.FONT_HERSHEY_DUPLEX,0.8,(55,255,255),1, cv2.LINE_AA)
    bins = len(hist)
    for i in range(bins):
        xcoord = int(i * cimadj.shape[1] / bins)
        ycoord = int(cimadj.shape[0]-hist[i]/hist.max()*cimadj.shape[0]/8)
        cv2.rectangle(cimadj, ( xcoord, cimadj.shape[0]),( xcoord+int(1/bins*cimadj.shape[1]), ycoord), (0,255,255), -1)
    if type(cimadj) is not np.ndarray:
        cimadj = cimadj.get()
    updateprog(100,f"displaying image: {image_counter}")
    while True:
        if not windowCreated:
            cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(windowName, 800,800)
            cv2.setWindowTitle(windowName, "Hologram Viewer - Press 'q' or ESC to quit")
            cv2.imshow(windowName, np.abs(cimadj))
            cv2.moveWindow(windowName, 0,0)
            windowCreated = True
        else:
            cv2.imshow(windowName, np.abs(cimadj))
        k=cv2.waitKey(1)
        if k == -1:
            break
        if k == ord('q') or k == 27:
            escaped = True
            break

        updateprog(100,f"waiting for user keyboard input")

        #print(f"{imgptr},{zi}", file=open(f"{datahome}/_curpos.csv", "w"))
        #os.rename(f"{datahome}/_curpos.csv", f"{datahome}/curpos.csv")
        # navigation keys
        #
        #   keyboard layout:
        # y u i      <-- higher z (by 100,10,1)
        # h j k      <-- lower z  (by 100,10,1)
        #      . ,   (next/prev image)
        # , - previous image
        """
        if k == ord(',') or k == 81 or k == 8:
            if imgptr > 1:
                imgptr -= 1
                break
        # . - next image
        if k == ord('.') or k == 83 or k==32:
            if imgptr < len(sys.argv)-1:
                imgptr += 1
                break
        """
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

    capok, capframe = cap.read()
    if capok:
        image_counter += 1
