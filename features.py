%matplotlib qt5
import PIL
from PIL import Image
from skimage.feature import local_binary_pattern, greycomatrix, greycoprops
from skimage.filters import gabor
import numpy as np
import pickle
import matplotlib.pyplot as plt
from pathlib import Path
import cv2
import os
import statistics as st
import csv
import math
measures = {'ar': [], 'per': [], 'Compactness': [], 'Deg. of Circ.': [], 'Contrast': [], 'Dissimilarity': [], 'Homogeneity': [], 'Energy': [], 'Correlation': [], 'm_gr': [], 'sd_gr': [], 'm_cl': [], 'sd_cl': [],'m_hue':[],'m_sat':[],'m_val':[],'sd_hue':[],'sd_sat':[],'sd_val':[]}
'''curr_dir=os.path.abspath(os.getcwd())
folders=os.listdir()'''
h_seg = os.listdir('C:/Users/Anurag/datasets/healthy_segmented')
m_seg = os.listdir('C:/Users/Anurag/datasets/ma_segmented')
for i in range(len(h_seg)):
    h_seg[i] = 'C:/Users/Anurag/datasets/healthy_segmented/' + h_seg[i]
for i in range(len(m_seg)):
    m_seg[i] = 'C:/Users/Anurag/datasets/ma_segmented/' + m_seg[i]
tot = h_seg + m_seg


#Area and permieter
for j in range(len(tot)):
    if(tot[j] != 'Thumbs.db'):
        img = cv2.imread(tot[j])
        im_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, thres = cv2.threshold(im_gray, 10, 255, 0)
        contours, hierarchy = cv2.findContours(thres, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        a = []
        p = []
        for k in range(len(contours)):
            area = cv2.contourArea(contours[k])
            perimeter = cv2.arcLength(contours[k], True)
            a.append(area)
            p.append(perimeter)
        if not a:
            measures['ar'].append(0)
        if not p:
            measures['per'].append(0)
        else:
            measures['ar'].append(st.mean(a))
            measures['per'].append(st.mean(p))
# GLCM features Contrast, Dissimilarity, Homogeneity, Energy, Correlation
for i in range(len(tot)):
    if(tot[i] != 'Thumbs.db'):
        img = Image.open(tot[i])
        img_gray = img.convert('L')  # Converting to grayscale
        img_arr = np.array(img_gray)
        gCoMat = greycomatrix(img_arr, [2], [0], 256, symmetric=True, normed=True)  # Co-occurance matrix
        contrast = greycoprops(gCoMat, prop='contrast')
        measures['Contrast'].append(contrast[0][0])
        dissimilarity = greycoprops(gCoMat, prop='dissimilarity')
        measures['Dissimilarity'].append(dissimilarity[0][0])
        homogeneity = greycoprops(gCoMat, prop='homogeneity')
        measures['Homogeneity'].append(homogeneity[0][0])
        energy = greycoprops(gCoMat, prop='energy')
        measures['Energy'].append(energy[0][0])
        correlation = greycoprops(gCoMat, prop='correlation')
        measures['Correlation'].append(correlation[0][0])
# Compactness and Deg. of Circ.

for i in range(len(measures['ar'])):
    if measures['ar'][i] == 0 and measures['per'][i] == 0:
        measures['Compactness'].append(0)
        measures['Deg. of Circ.'].append(0)
    elif measures['ar'][i] == 0 or measures['per'][i] == 0:
        measures['Compactness'].append(0)
        measures['Deg. of Circ.'].append(0)
    else:
        temp = measures['per'][i]**2 / (4 * math.pi * measures['ar'][i])
        measures['Compactness'].append(temp)
        measures['Deg. of Circ.'].append(1 / temp)

# mean and standard deviation of green channel pixels in original and CLAHE images
h_seg = os.listdir('C:/Users/Anurag/datasets/healthy_segmented')
m_seg = os.listdir('C:/Users/Anurag/datasets/ma_segmented')
for i in range(len(h_seg)):
    img = cv2.imread('C:/Users/Anurag/datasets/healthy_segmented/' + h_seg[i])
    im=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    hsv=cv2.cvtColor(im,cv2.COLOR_RGB2HSV)
    h,s,v=cv2.split(hsv)
    im_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    threshold_level = 10
    coords = np.column_stack(np.where(im_gray >= threshold_level))
    org = cv2.imread('C:/Users/Anurag/datasets/Healthy/' + h_seg[i][-12:-4] + '.jpg')
    b, green_fundus, r = cv2.split(org)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrast_enhanced_green_fundus = clahe.apply(green_fundus)
    m = [int(green_fundus[coords[j][0]][coords[j][1]]) for j in range(len(coords))]
    m_cla = [int(contrast_enhanced_green_fundus[coords[j][0]][coords[j][1]]) for j in range(len(coords))]
    hue=[int(h[coords[j][0]][coords[j][[1]]]) for j in range(len(coords))]
    sat=[int(s[coords[j][0]][coords[j][[1]]]) for j in range(len(coords))]
    val=[int(v[coords[j][0]][coords[j][[1]]]) for j in range(len(coords))]
    if not m:
        measures['sd_gr'].append(0)
        measures['m_gr'].append(0)
        measures['m_hue'].append(0)
        measures['sd_hue'].append(0)
        measures['m_sat'].append(0)
        measures['sd_sat'].append(0)
        measures['m_val'].append(0)
        measures['sd_val'].append(0)
    if not m_cla:
        measures['sd_cl'].append(0)
        measures['m_cl'].append(0)
    else:
        temp=st.mean(m)
        temp_1=st.mean(m_cla)
        temp_2=st.mean(hue)
        temp_3=st.mean(val)
        temp_4=st.mean(sat)
        sd = [(m[j] - temp)**2 for j in range(len(m))]
        sd_cla = [(m_cla[j] - temp_1)**2 for j in range(len(m))]
        sd_h=[(hue[j]-temp_2)**2 for j in range(len(hue))]
        sd_v=[(val[j]-temp_3)**2 for j in range(len(val))]
        sd_s=[(sat[j]-temp_4)**2 for j in range(len(sat))]
        measures['sd_gr'].append(math.sqrt(st.mean(sd)))
        measures['sd_cl'].append(math.sqrt(st.mean(sd_cla)))
        measures['m_gr'].append(st.mean(m))
        measures['m_cl'].append(st.mean(m_cla))
        measures['m_hue'].append(st.mean(hue))
        measures['sd_hue'].append(math.sqrt(st.mean(sd_h)))
        measures['m_val'].append(st.mean(val))
        measures['sd_val'].append(math.sqrt(st.mean(sd_v)))
        measures['m_sat'].append(st.mean(sat))
        measures['sd_sat'].append(math.sqrt(st.mean(sd_s)))

for i in range(len(m_seg)):
    img = cv2.imread('C:/Users/Anurag/datasets/healthy_segmented/' + h_seg[i])
    im=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    hsv=cv2.cvtColor(im,cv2.COLOR_RGB2HSV)
    h,s,v=cv2.split(hsv)
    im_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    threshold_level = 10
    coords = np.column_stack(np.where(im_gray >= threshold_level))
    org = cv2.imread('C:/Users/Anurag/datasets/Healthy/' + h_seg[i][-12:-4] + '.jpg')
    b, green_fundus, r = cv2.split(org)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrast_enhanced_green_fundus = clahe.apply(green_fundus)
    m = [int(green_fundus[coords[j][0]][coords[j][1]]) for j in range(len(coords))]
    m_cla = [int(contrast_enhanced_green_fundus[coords[j][0]][coords[j][1]]) for j in range(len(coords))]
    hue=[int(h[coords[j][0]][coords[j][[1]]]) for j in range(len(coords))]
    sat=[int(s[coords[j][0]][coords[j][[1]]]) for j in range(len(coords))]
    val=[int(v[coords[j][0]][coords[j][[1]]]) for j in range(len(coords))]
    if not m:
        measures['sd_gr'].append(0)
        measures['m_gr'].append(0)
        measures['m_hue'].append(0)
        measures['sd_hue'].append(0)
        measures['m_sat'].append(0)
        measures['sd_sat'].append(0)
        measures['m_val'].append(0)
        measures['sd_val'].append(0)
    if not m_cla:
        measures['sd_cl'].append(0)
        measures['m_cl'].append(0)
    else:
        temp=st.mean(m)
        temp_1=st.mean(m_cla)
        temp_2=st.mean(hue)
        temp_3=st.mean(val)
        temp_4=st.mean(sat)
        sd = [(m[j] - temp)**2 for j in range(len(m))]
        sd_cla = [(m_cla[j] - temp_1)**2 for j in range(len(m))]
        sd_h=[(hue[j]-temp_2)**2 for j in range(len(hue))]
        sd_v=[(val[j]-temp_3)**2 for j in range(len(val))]
        sd_s=[(sat[j]-temp_4)**2 for j in range(len(sat))]
        measures['sd_gr'].append(math.sqrt(st.mean(sd)))
        measures['sd_cl'].append(math.sqrt(st.mean(sd_cla)))
        measures['m_gr'].append(st.mean(m))
        measures['m_cl'].append(st.mean(m_cla))
        measures['m_hue'].append(st.mean(hue))
        measures['sd_hue'].append(math.sqrt(st.mean(sd_h)))
        measures['m_val'].append(st.mean(val))
        measures['sd_val'].append(math.sqrt(st.mean(sd_v)))
        measures['m_sat'].append(st.mean(sat))
        measures['sd_sat'].append(math.sqrt(st.mean(sd_s)))


with open("C:\\Users\\Anurag\\Desktop\\newfeatures.csv", "w") as outfile:  # "a" for appending values to csv
    writer = csv.writer(outfile)
    writer.writerow(measures.keys())
    writer.writerows(zip(*measures.values()))
