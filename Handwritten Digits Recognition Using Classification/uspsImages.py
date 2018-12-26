from PIL import Image
import glob
import numpy as np

def preprocess():
	image_list = []
	labels_list=[]
	for i in range(0,10):
	    cnt=0
	    for filename in glob.glob('Numerals/'+str(i)+'/*.png'):
	        im=Image.open(filename)
	        im=im.resize((28,28))
	        temp1=list(im.getdata())
	        temp2=[]
	        for j in temp1:
	            temp2.append((255-j)/255)
	        image_list.append(temp2)
	        im.close()
	        cnt=cnt+1
	    for j in range(0,cnt):
	        temp1=[]
	        for k in range(0,10):
	            temp1.append(0)
	        temp1[i]=1
	        labels_list.append(temp1)
	images=np.array(image_list)
	labels=np.array(labels_list)
	return images,labels