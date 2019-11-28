import numpy as np
from matplotlib.pyplot import imshow, show, colorbar

from PIL import Image

input_file = './2019-11-27 16:33:00/images.txt'

f=open(input_file, "r")
# a = np.load(f)
flat_images =f.readlines()
i = 0
for x in flat_images:
	i+= 1
	if i < 2000:
		continue
	flat_img = np.array(x.split()).astype(np.float)
	img = flat_img.reshape(3,84,84).swapaxes(0,1).swapaxes(1,2)
	im = Image.fromarray(((img - np.min(img))*255/(np.max(img - np.min(img)))).astype(np.uint8))
	# right now i save all the images to a single file, which means i only save 1 image lol:
	im.save('testimg.png')
	# these look pretty bad for some reason:
	imshow(img)
	show()