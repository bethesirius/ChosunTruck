#A tool for hand labelling images
#Generates an IDL file
#pass in the directory where you store your images and a filename, then select the points on the images
#every time you hit next a line is generated
#the clear button removes are selected points on the current image
#when all files in the directory are processed, the idl file is written out

#ex: python make_idl.py train640x480 train.idl

import sys
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib as mpl
import matplotlib.patches as patches
mpl.rcParams['toolbar'] = 'None'
from matplotlib.widgets import Button
from os import listdir
from os.path import isfile, join

top_corners = []
bottom_corners = []
patchCache = [] #the rectangles that get drawn on the image, stored so they can be removed in an orderly fashion

def removeAllPatches():
    for patch in patchCache:
        patch.remove()
    patchCache[:] = []

def next(event):  #called when the next button is hit
    global filename
    line = []
    line.append('"' + filename + '": ')
    one_decimal = "{0:0.1f}"
    for i in range(len(top_corners)):
        line.append('(' + one_decimal.format(top_corners[i][0]) + ', ' + one_decimal.format(top_corners[i][1]) + ', ' + one_decimal.format(bottom_corners[i][0])  + ', ' + one_decimal.format(bottom_corners[i][1]) + ')')
        if i != len(top_corners) - 1:
            line.append(',')
        else:
            line.append(';' + "\n")
    text_line = ''.join(line)
    outfile.write(text_line)
    print "writing line : " + text_line
    if len(onlyfiles) == 0:
        plt.close()
    else:
        filename = path + "/" + onlyfiles.pop()
        image = mpimg.imread(filename)
        imshow_obj.set_data(image)
        top_corners[:] = []
        bottom_corners[:] = []
        removeAllPatches()
    

def clear(event): #called when the clear button is hit
    top_corners[:] = []
    bottom_corners[:] = []
    removeAllPatches()

def onclick(event):  #called when anywhere inside the window is clicked
    if event.xdata > 1 and event.ydata > 1:
        if (len(top_corners) > len(bottom_corners)):
            bottom_corners.append([event.xdata,event.ydata])
            patchCache.append(patches.Rectangle((top_corners[-1][0], top_corners[-1][1])
                                           ,bottom_corners[-1][0] - top_corners[-1][0], bottom_corners[-1][1] - top_corners[-1][1],
                                           hatch='/',fill=False))
            ax.add_patch(patchCache[-1])
            plt.draw()
        else:
            top_corners.append([event.xdata,event.ydata])

ax = plt.gca()

#get our files for processing
if len(sys.argv) < 3:
    print "Too few params, try something like:  python make_idl.py train640x480 train.idl"
    exit()
path = sys.argv[1]
outfile_name = sys.argv[2]
outfile = open(outfile_name, 'w')
onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]

#
filename = path + "/" + onlyfiles.pop()
image = mpimg.imread(filename)
imshow_obj = ax.imshow(image)

plt.axis("off")
fig = plt.gcf()
fig.canvas.mpl_connect('button_press_event', onclick)

#add the buttons to the bottom of the window
axnext = plt.axes([0.7, 0.01, 0.1, 0.075])
axclear = plt.axes([0.81, 0.01, 0.1, 0.075])
bnext = Button(axnext, 'Next')
bnext.on_clicked(next)
bclear = Button(axclear, 'Clear')
bclear.on_clicked(clear)
plt.show()

outfile.close()

print "finished"
