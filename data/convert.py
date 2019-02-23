import sys
import numpy as np
import PIL
from numpy import load
from PIL import Image, ImageSequence

#validate arguments
if len(sys.argv) != 4:
	print('Incorrect number of arguments. (Input_volume Input_label Output)')
	exit();
input_volume_file_name = str(sys.argv[1])
input_label_file_name = str(sys.argv[2])

output_file_name = str(sys.argv[3])

print('Opening inputs')
#Get the data from the .tiff format
im_vol = Image.open(input_volume_file_name)
im_label = Image.open(input_label_file_name)

print('Converting to np')
imarray_vol = []
imarray_label = []
for i in range(30):
	try:
		im_vol.seek(i)
		im_label.seek(i)
		imarray_vol.append(np.array(im_vol))
		imarray_label.append(np.array(im_vol))
	except EOFError:
		# Not enough frames in img
		break

print(len(imarray_vol))
print(len(imarray_label))
print('Saving as .npz')
#save it in the format of .npz
np.savez(output_file_name, volume=imarray_vol,label=imarray_label)
print('Done')
