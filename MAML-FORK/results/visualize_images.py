import numpy as np
from matplotlib.pyplot import imshow, show, colorbar
from matplotlib import pyplot

from PIL import Image


input_file = './2019-11-28 17:38:00/images_step0.txt'

f=open(input_file, "r")
# a = np.load(f)
flat_images =f.readlines()
images_arr = []
i = 0
for x in flat_images:	
	flat_img = np.array(x.split()).astype(np.float)
	img = flat_img.reshape(3,84,84).swapaxes(0,1).swapaxes(1,2)
	im = Image.fromarray(((img - np.min(img))*255/(np.max(img - np.min(img)))).astype(np.uint8))
	# right now i save all the images to a single file, which means i only save 1 image lol:
	# want to figure out how to tile them to a file eventually
	images_arr.append(im)
	pil_im = np.asarray(im)
	if i < 49:
		pyplot.subplot(7, 7, 1 + i)
		pyplot.axis('off')
		pyplot.imshow(pil_im)
	i += 1
pyplot.savefig(input_file[:-4]+".png")
# these look pretty bad for some reason:
# imshow(img)
# show()