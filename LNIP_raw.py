
#Taking the input for pixels of window size 3
i6,i7,i8 = input().split()
i5,Ic,i1 = input().split()
i4,i3,i2 = input().split()

#lists for storing the sign and magnitude patterns
signs = []
magn = []


#Storing the neighbours of each pixel
nei_i1 = [i7,i8,i2,i3]
nei_i2 = [i1,i3]
nei_i3 = [i1,i2,i4,i5]
nei_i4 = [i3,i5]
nei_i5 = [i3,i4,i6,i7]
nei_i6 = [i5,i7]
nei_i7 = [i5,i6,i8,i1]
nei_i8 = [i7,i1]
all_nei = [i1,i2,i3,i4,i5,i6,i7,i8]

#Making dictonary function for easy access
indices = {'i1':i1,'i2':i2,'i3':i3,'i4':i4,'i5':i5,'i6':i6,'i7':i7,'i8':i8}
neigh_lists  = {'ne_i1':nei_i1,'ne_i2':nei_i2,'ne_i3':nei_i3,'ne_i4':nei_i4,'ne_i5':nei_i5,'ne_i6':nei_i6,'ne_i7':nei_i7,'ne_i8':nei_i8}

#Function for calculating the B(1,i)
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

#Function for calculating the B(2,i)
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

#Function for calculating the Magnitude
def mags(neis,comp):
    m_sum = 0.0
    for k in neis:
        m_sum+=abs((int(k)-int(comp)))
    return float(m_sum/len(neis))

#Function for calculating the Threshold
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

print ('Sign Code is ->  '+''.join(signs)+'  Magnitude Code is ->  '+''.join(magn)) #Prints the magnitude and Sign pattern of the input matrix

# res1 = B_1_i(nei_i1,i1)
# res2 = B_2_i(nei_i1,Ic)
# res3 = int(res1,2)^int(res2,2)
# D = bin(res3)[2:].count('1')
# M = len(nei_i1)
# if(D>=int((M/2))):
#     I1 = 1
# else:
#     I1 = 0

#print res1
#print res2