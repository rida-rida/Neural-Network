


import coco_text
import cv2
import os
ct = coco_text.COCO_Text('cocotext.v2.json')
ct.info()
imgs = ct.getImgIds(imgIds=ct.val, catIds=[('legibility','legible'),('class','machine printed')])
anns = ct.getAnnIds(imgIds=ct.val, catIds=[('legibility','legible'),('class','machine printed')], areaRng=[0,200])


dataDir='/Data3/data'
dataType='train2014'




import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
pylab.rcParams['figure.figsize'] = (10.0, 8.0)


# get all images containing at least one instance of legible text
imgIds = ct.getImgIds(imgIds=ct.val, catIds=[('legibility','legible')])

#print(imgIds)
#print(len(imgIds))
# pick one at random
path='/Data3/coco_val1/'
for x in range(len(imgIds)):
	img = ct.loadImgs(imgIds[np.random.randint(0,len(imgIds))])[0]
	I = io.imread('%s/%s/%s'%(dataDir,dataType,img['file_name']))
	plt.imshow(I)
	
	#lt.show()
	annIds = ct.getAnnIds(imgIds=img['id'])
	anns = ct.loadAnns(annIds)
	ct.showAnns(anns,show_mask=True)
	#plt.show()
	plt.axis('off')
	plt.savefig('/Data3/coco_val2/'+str(img['file_name'])+'.png',bbox_inches='tight')
	plt.close()
	#plt.show()
	#plt.show()
	#cv2.imwrite(os.path.join(path , img['file_name']), I)
#print (dataType,img['file_name'])
#print(I)
#print(dataType,img['file_name'])
#cv2.imshow('x',I)  
#cv2.waitKey(0)
#plt.plot(I)
#plt.show()
#plt.figure()
#plt.imshow(I)
#plt.show()
# load and display text annotations
#cv2.imshow('x',I)
#cv2.waitKey(0)
#path='/Data3/coco_val/'
#cv2.imwrite(os.path.join(path , img['file_name']), I)
