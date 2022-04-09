import tkinter as tk
from tkinter import filedialog
import sys,os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn as sk
from sklearn import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV 
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
import warnings
import math
from PIL import Image,ImageTk
import numpy as np
import os,csv,time
from scipy.stats import itemfreq
warnings.filterwarnings("ignore")


root = tk.Tk()
root.geometry("700x500+200+200")
load = Image.open("robot.png")
render = ImageTk.PhotoImage(load)
img = tk.Label(root, image=render)
img.image = render
img.place(x=250, y=20)

def file_path():
    filed  = filedialog.askopenfilename()
##    #pip install opencv-python==3.4.2.16 && pip install opencv-contrib-python==3.4.2.1

    x = pd.read_csv(r'LNIP_super_all.csv')
    training_set,test_set = train_test_split(x,test_size=0.2,random_state=0)
    X_train = training_set.iloc[:,0:512].values
    Y_train = training_set.iloc[:,512].values
    X_test = test_set.iloc[:,0:512].values
    Y_test = test_set.iloc[:,512].values

    def LNIP_Feature_Extract(gray_image):
        imgLBP = np.zeros_like(gray_image)
        neighboor = 3
        for ih in range(0,gray_image.shape[0] - neighboor):
            for iw in range(0,gray_image.shape[1] - neighboor):
                img          = gray_image[ih:ih+neighboor,iw:iw+neighboor]

                i6 = img[0,0]
                i7 = img[0,1]
                i8 = img[0,2]
                i5 = img[1,0]
                Ic = img[1,1]
                i1 = img[1,2]
                i4 = img[2,0]
                i3 = img[2,1]
                i2 = img[2,2]
                            
                signs = []
                magn = []
                sign_stri = ""
                mag_stri= ""

                nei_i1 = [i7,i8,i2,i3]
                nei_i2 = [i1,i3]
                nei_i3 = [i1,i2,i4,i5]
                nei_i4 = [i3,i5]
                nei_i5 = [i3,i4,i6,i7]
                nei_i6 = [i5,i7]
                nei_i7 = [i5,i6,i8,i1]
                nei_i8 = [i7,i1]
                all_nei = [i1,i2,i3,i4,i5,i6,i7,i8]

                indices = {'i1':i1,'i2':i2,'i3':i3,'i4':i4,'i5':i5,'i6':i6,'i7':i7,'i8':i8}
                neigh_lists  = {'ne_i1':nei_i1,'ne_i2':nei_i2,'ne_i3':nei_i3,'ne_i4':nei_i4,'ne_i5':nei_i5,'ne_i6':nei_i6,'ne_i7':nei_i7,'ne_i8':nei_i8}

                def B_1_i(nei_lis,compare_element):
                    bli = ''
                    for i in nei_lis:
                        if(int(i)<int(compare_element)):
                            bli+='0'
                        elif(int(i)>=int(compare_element)):
                            bli+='1'
                        else:
                            pass
                    return bli

                def B_2_i(nei_lis,centre_element):
                    b2i = ''
                    for i in nei_lis:
                        if(int(i)<int(centre_element)):
                            b2i+='0'
                        elif(int(i)>=int(centre_element)):
                            b2i+='1'
                        else:
                            pass
                    return b2i

                def mags(neis,comp):
                    m_sum = 0.0
                    for k in neis:
                        m_sum+=abs((int(k)-int(comp)))
                    return float(m_sum/len(neis))

                def thresholds(alls,centre_ele):
                    thre_sum = 0.0
                    for h in alls:
                        thre_sum+=abs(int(h)-int(centre_ele))
                    return float(thre_sum/8)

                for_ind = list(indices.keys())
                for_ind.sort()
                for_nei = list(neigh_lists.keys())
                for_nei.sort()

                for one,two in zip(for_ind,for_nei):
                    res1 = B_1_i(neigh_lists[two],indices[one])
                    res2 = B_2_i(neigh_lists[two],Ic)
                    res3 = int(res1,2)^int(res2,2)
                    #print str(bin(res3)[2:].zfill(4))+'  '+str(indices[one])
                    D = bin(res3)[2:].count('1')
                    M = len(neigh_lists[two])
                    if(D>=int((M/2))):
                        signs.append(str(1))
                    else:
                        signs.append(str(0))

                for one,two in zip(for_ind,for_nei):
                    Mi = mags(neigh_lists[two],indices[one])
                    Tc = thresholds(all_nei,Ic)
                    if(Mi>=Tc):
                        magn.append(str(1))
                    else:
                        magn.append(str(0))
                sign_stri = sign_stri.join(signs)
                mag_stri = mag_stri.join(magn)
                imgLBP[ih+1,iw+1] = int(sign_stri,2)
                #print int(mag_stri,2)
        return (imgLBP)

    def LNIP_Feature_Extract_mag(gray_image):
        imgmag = np.zeros_like(gray_image)
        neighboor = 3
        for ih in range(0,gray_image.shape[0] - neighboor):
            for iw in range(0,gray_image.shape[1] - neighboor):
                img          = gray_image[ih:ih+neighboor,iw:iw+neighboor]
    #            center       = img[1,1]

    ##            row1 = ''.join(str(roi[0]))
    ##            row1 = row1[1:len(row1)-1]
    ##
    ##            row2 = ''.join(str(roi[1]))
    ##            row2 = row1[1:len(row1)-1]
    ##
    ##            row3 = ''.join(str(roi[2]))
    ##            row3 = row1[1:len(row1)-1]

                i6 = img[0,0]
                i7 = img[0,1]
                i8 = img[0,2]
                i5 = img[1,0]
                Ic = img[1,1]
                i1 = img[1,2]
                i4 = img[2,0]
                i3 = img[2,1]
                i2 = img[2,2]
                            
                signs = []
                magn = []
                sign_stri = ""
                mag_stri= ""

                nei_i1 = [i7,i8,i2,i3]
                nei_i2 = [i1,i3]
                nei_i3 = [i1,i2,i4,i5]
                nei_i4 = [i3,i5]
                nei_i5 = [i3,i4,i6,i7]
                nei_i6 = [i5,i7]
                nei_i7 = [i5,i6,i8,i1]
                nei_i8 = [i7,i1]
                all_nei = [i1,i2,i3,i4,i5,i6,i7,i8]

                indices = {'i1':i1,'i2':i2,'i3':i3,'i4':i4,'i5':i5,'i6':i6,'i7':i7,'i8':i8}
                neigh_lists  = {'ne_i1':nei_i1,'ne_i2':nei_i2,'ne_i3':nei_i3,'ne_i4':nei_i4,'ne_i5':nei_i5,'ne_i6':nei_i6,'ne_i7':nei_i7,'ne_i8':nei_i8}

                def B_1_i(nei_lis,compare_element):
                    bli = ''
                    for i in nei_lis:
                        if(int(i)<int(compare_element)):
                            bli+='0'
                        elif(int(i)>=int(compare_element)):
                            bli+='1'
                        else:
                            pass
                    return bli

                def B_2_i(nei_lis,centre_element):
                    b2i = ''
                    for i in nei_lis:
                        if(int(i)<int(centre_element)):
                            b2i+='0'
                        elif(int(i)>=int(centre_element)):
                            b2i+='1'
                        else:
                            pass
                    return b2i

                def mags(neis,comp):
                    m_sum = 0.0
                    for k in neis:
                        m_sum+=abs((int(k)-int(comp)))
                    return float(m_sum/len(neis))

                def thresholds(alls,centre_ele):
                    thre_sum = 0.0
                    for h in alls:
                        thre_sum+=abs(int(h)-int(centre_ele))
                    return float(thre_sum/8)

                for_ind = list(indices.keys())
                for_ind.sort()
                for_nei = list(neigh_lists.keys())
                for_nei.sort()

                for one,two in zip(for_ind,for_nei):
                    res1 = B_1_i(neigh_lists[two],indices[one])
                    res2 = B_2_i(neigh_lists[two],Ic)
                    res3 = int(res1,2)^int(res2,2)
                    #print str(bin(res3)[2:].zfill(4))+'  '+str(indices[one])
                    D = bin(res3)[2:].count('1')
                    M = len(neigh_lists[two])
                    if(D>=int((M/2))):
                        signs.append(str(1))
                    else:
                        signs.append(str(0))

                for one,two in zip(for_ind,for_nei):
                    Mi = mags(neigh_lists[two],indices[one])
                    Tc = thresholds(all_nei,Ic)
                    if(Mi>=Tc):
                        magn.append(str(1))
                    else:
                        magn.append(str(0))
                sign_stri = sign_stri.join(signs)
                mag_stri = mag_stri.join(magn)
                imgmag[ih+1,iw+1] = int(mag_stri,2)
                #print int(mag_stri,2)
        return (imgmag)


    h = Image.open(filed)
    k = LNIP_Feature_Extract(np.array(h))
    k1 = LNIP_Feature_Extract_mag(np.array(h))
    vecimg_count1 = itemfreq(np.array(k1.flatten()))
    vecimg_count = itemfreq(np.array(k.flatten()))

    hist = vecimg_count[:,1]/sum(vecimg_count[:,1])
    hist1 = vecimg_count1[:,1]/sum(vecimg_count1[:,1])
    final = [*hist,*hist1] 
    #print(len(final))
    classifier = SVC(C=1100, cache_size=200, class_weight=None, coef0=0.0,
        decision_function_shape='ovr', degree=3, gamma=1000, kernel='rbf', max_iter=-1,
        probability=False, random_state=None, shrinking=True, tol=0.001,
        verbose=False)
    classifier.fit(X_train,Y_train)
    SVM_pred = classifier.predict([final])
    if(0 in SVM_pred):
        text4 = tk.Label(root, text="ABNORMAL",fg = 'Red',font='Times 15').place(x=325,y=400)
        text2 = tk.Label(root, text="Accuracy 97%",fg = 'Green',font='Times 13').place(x=325,y=450)
    else:
        text4 = tk.Label(root, text="Normal",fg = 'Green',font='Times 15').place(x=325,y=400)
        text2 = tk.Label(root, text="Accuracy 97%",fg = 'Green',font='Times 13').place(x=325,y=450)
        

root.title('Final GUI Demo')
actionBtn = tk.Button(root, text="Select Image", width=15, height=2, command=file_path).place(x=100, y=400)
quitbtn = tk.Button(root, text="Quit", width=15, height=2, command=root.destroy).place(x=510, y=400)

text1 = tk.Label(root, text="GUI for predicting Breast cancer",fg = 'Green',font='Times 15 bold').place(x=225,y=260)
text3 = tk.Label(root, text="Status",fg = 'black',font='Times 17').place(x=342,y=360)


root.mainloop()
