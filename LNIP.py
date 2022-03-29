import time
import csv,os
import cv2
import numpy as np
from sklearn import linear_model
from skimage.transform import integral_image
from skimage.feature import draw_multiblock_lbp
from skimage. feature import hog
from skimage.feature import local_binary_pattern,multiblock_lbp
from scipy.stats import itemfreq
import warnings
warnings.filterwarnings("ignore")

normal_dirs = (r'/Users/vikasreddy/test1/cancer/Dataset_DDSM_database/Normal')
abnormal_dirs = (r'/Users/vikasreddy/test1/cancer/Dataset_DDSM_database/Abnormal')
nl = os.listdir(normal_dirs)
abl = os.listdir(abnormal_dirs)

def lbp_histreturn(m):
    img = cv2.imread(m)
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #sift = cv2.xfeatures2d.SIFT_create()
    #kp,des = sift.detectAndCompute(gray,None)

    #kp = sift.detect(gray,None)
    #img=cv2.drawKeypoints(gray,kp,img)
    #cv2.imwrite('sift_keypoints.jpg',img)

    #surf = cv2.xfeatures2d.SURF_create()
    #kp,des = surf.detectAndCompute(gray,None)
    #img = cv2.drawKeypoints(gray,kp,img)
    #cv2.imwrite('surf_keypoints.jpg',img)
    
    radius =1
    n_points =8*radius
    METHOD = 'nri_uniform'
    lbp =local_binary_pattern(gray, n_points, radius, METHOD)
    x = itemfreq(lbp.ravel())
    hist=x[:,1]/sum(x[:,1])
    return hist

with open("normal.csv","a",newline='') as fp:
    wr=csv.writer(fp,dialect='excel')
    for i in nl:
        k = lbp_histreturn(r'/Users/vikasreddy/test1//cancer/Dataset_DDSM_database/Normal'+i)
        wr.writerow(k)

print("Completed Normal")


        
with open("abnormal.csv","a",newline='') as fp:
    wr=csv.writer(fp,dialect='excel')
    for i in abl:
        k = lbp_histreturn(r'/Users/vikasreddy/test1/cancer/Dataset_DDSM_database/Abnormal'+i)
        wr.writerow(k)
        
print("Completed Abnormal")

