from liblinearutil import *
import os, sys, csv
import numpy as np
from sklearn import svm
from sklearn import preprocessing
def classification(train_range, test_range):
   label=[[4,1,3,2,0,4,1,3,2,0,4,1,3,2,0],[2,1,3,0,4,4,0,3,2,1,3,4,1,2,0],[2,1,3,0,4,4,0,3,2,1,3,4,1,2,0]]    
   cm=np.zeros(shape=(5,5),dtype=float)
   eyeDir = 'seed_5_eye_all_feature_npy/'
   fileNames = os.listdir(eyeDir)
   fileNames.sort()
   for filename in fileNames:
       eyeSubDir=os.listdir(eyeDir+filename+'/')
       eyeSubDir.sort()
       sum_train=[]
       sum_test=[]
       train_label=[]
       count=0
       test_label=[]  
       for subfilename in eyeSubDir:
           l=int(subfilename) 
           file_path1=eyeDir+filename+'/'+subfilename
           for i in train_range:
               pretrain11=np.load(file_path1+'/'+"eye_"+str(i)+".npy")
               pretrain1=preprocessing.scale(pretrain11)
               pretrain=pretrain1.transpose()
               cSize=pretrain11.shape[1]
               for ii in range(cSize):
                    sum_train.append(pretrain[ii])
                    train_label.append(label[l-1][i-1])
           for j in test_range:
               pretest11=np.load(file_path1+'/'+"eye_"+str(j)+".npy")
               pretest1=preprocessing.scale(pretest11)
               pretest=pretest1.transpose()
               tSize=pretest11.shape[1]
               for ii in range(tSize):
                    sum_test.append(pretest[ii])
                    count=count+1
                    test_label.append(label[l-1][j-1])
       
       max=0.0     
       for o in np.linspace(1,1024,20):
               prob = problem(train_label, sum_train)
               params=('-s 2 -c'+' '+str(o))
               model=train(prob,params)
               a,b,c=predict(test_label,sum_test,model,'-q')
               if b[0]>max:
                   max=b[0]
                   best_pre=a
       for o in np.linspace(0.001,1,20):
               prob = problem(train_label, sum_train)
               params=('-s 3 -c'+' '+str(o))
               model=train(prob,params)
               a,b,c=predict(test_label,sum_test,model,'-q')
               if b[0]>max:
                   max=b[0]
                   best_pre=a            
       print max,best_pre
       for number in range(count):
            x=int(test_label[number])
            y=int(best_pre[number])
            cm[x][y]=cm[x][y]+1
   print np.mat(cm)
   return cm

if __name__ == '__main__':
     cm1=classification(train_range=range(1,11), test_range=range(11,16))  
     cm2=classification(train_range=range(1,6)+range(11,16), test_range=range(6,11)) 
     cm3=classification(train_range=range(6,16), test_range=range(1,6)) 
     cm_mean=cm1+cm2+cm3
     cm_normalized = cm_mean.astype('float') / cm_mean.sum(axis=1)[:, np.newaxis]
     print cm_normalized
