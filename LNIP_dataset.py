from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import time
import csv,os
import cv2
from sklearn import linear_model
from skimage.transform import integral_image
from skimage.feature import draw_multiblock_lbp
from skimage. feature import hog
from skimage.feature import local_binary_pattern,multiblock_lbp
from scipy.stats import itemfreq
import warnings
warnings.filterwarnings("ignore")

def file_path():
    filed  = filedialog.askopenfilename()
    normal_dirs = (r'//Users//vikasreddy//test1/cancer//Dataset_DDSM_database//Normal')
    abnormal_dirs = (r'//Users//vikasreddy//test1/cancer//Dataset_DDSM_database//Abnormal')
    nl = os.listdir(normal_dirs)
    abl = os.listdir(abnormal_dirs)

    def LNIP_Feature_Extract(gray_image):
        imgLBP = np.zeros_like(gray_image)
        neighboor = 3
        for ih in range(0,gray_image.shape[0] - neighboor):
            for iw in range(0,gray_image.shape[1] - neighboor):
                img          = gray_image[ih:ih+neighboor,iw:iw+neighboor]
                center       = img[1,1]

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
                imgLBP[ih+1,iw+1] = int(mag_stri,2)
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

with open("normal.csv","a",newline='') as fp:
    wr=csv.writer(fp,dialect='excel')
    for i in nl:
        k = LNIP_Feature_Extract(r'//Users//vikasreddy//test1//cancer//Dataset_DDSM_database//Normal'+i)
        wr.writerow(k)

print("Completed Normal")


        
with open("abnormal.csv","a",newline='') as fp:
    wr=csv.writer(fp,dialect='excel')
    for i in abl:
        k = LNIP_Feature_Extract(r'//Users//vikasreddy//test1//cancer//Dataset_DDSM_database//Abnormal'+i)
        wr.writerow(k)
        
print("Completed Abnormal")