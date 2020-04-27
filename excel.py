import glob
import cv2
import csv
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.metrics import accuracy_score




class ConvertImageAndKNN:

	def __init__(self):

		self.glob_list=[];


	def image_resizing(self):
		pass

	def image_menipulation_into_grayscale(self):
		pass

	def load_dataSet(self):


		b=[x for x in glob.glob('DevanagariHandwrittenCharacterDataset/Train/*')]

		#import pdb;pdb.set_trace()
		j=0 
		fin={}
		final_data=[]
		final_value=[]
		while j<len(b):
			y=glob.glob(b[j]+'/*')
			fin[j]=b[j].split("/")[-1]
			print(j)
			k=0
			new=[]
			val=[]
			while k<len(y):
				data=cv2.imread(y[k],0)
				print(k)
				new.append(data.ravel())
				val.append(j)
				k=k+1
			final_data.extend(new)
			final_value.extend(val)
			j=j+1
		test=final_data[0:1700:60]
		label=final_value[0:1700:60]
		self.glob_list=[final_data,final_value,test,label]
		return self.glob_list



	def add_knn(self):

		knn=KNeighborsClassifier()
		#import pdb;pdb.set_trace()
		x=self.load_dataSet()
		knn.fit(x[0],x[1])
		#import pdb;pdb.set_trace()
		pridict=knn.predict(x[2])
		percatage=accuracy_score(x[3],pridict)
		print(percatage)



obj=ConvertImageAndKNN()
obj.add_knn()


'''print(b)
i=0
j={}
bata=[]
dalue=[]
alldata=[]
print(len(a))
while i<len(a):
	b=cv2.imread(a[i],0)
	bata.append(b.ravel())
	dalue.append(1)
	i=i+1
#import pdb;pdb.set_trace()
dalue[-1]=0
value=np.array(dalue)[:279]
data=np.array(bata)[:279]
test=bata[279:300]
test_label=dalue[279:300]

alldata.append(value)
alldata.append(data)
#import pdb;pdb.set_trace()

'''
	



