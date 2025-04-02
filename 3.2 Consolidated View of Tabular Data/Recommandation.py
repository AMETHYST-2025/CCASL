import os
from tabulate import tabulate
import shutil
import cv2
import sklearn
import numpy as np
import os
import layoutparser as lp
from matplotlib import pyplot as plt
from tqdm import tqdm
import pytesseract
import pandas as pd
import requests
import nltk
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
np.random.seed(42)
import shutil
import easyocr
import matplotlib.pyplot as plt


path = os.getcwd()

def plt_imshow(title, image):
    # convert the image frame BGR to RGB color space and display it
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	plt.imshow(image)
	plt.title(title)
	plt.grid(False)
	plt.show()


def searchInName(text, key):
    occurrences = 0
    presence = 0
    # Title = to_lower(Title)
    tokens = word_tokenize(text)
    punctuation = ['(', ')', ';', ':', '[', ']', ',','-', '/','_']
    stop_words = stopwords.words('english')
    keywords = [word for word in tokens if not word in stop_words and not word in punctuation]
    for k in keywords:
        if key.lower() in k.lower(): occurrences += 1  # k.lower(), for low letter
    if occurrences != 0:
        presence = 1;  #if the searched word exists in the text, "Presence" takes the value 1, and 0 otherwise.
    return occurrences, presence

key_words = ["Sample", "Tg", "Storage", "Modulus", "Tensile", "E’", "G’", "Rubbery", "Glassy", "MPa", "GPa", "°C", "°K"]

def pertinence(text, image_name, key_words = key_words):
    # cleaning the text, removing punctuation
    text = re.sub('\W+\s*', ' ', text)
    text = re.sub('_', ' ', text)
    Interest = 0
    result = []
    line_DF = {}
    line_DF['Ref']= image_name
    for search_for in key_words:  # Select text containing keywords, with the "Interest" parameter
        occ, pres = searchInName(text, search_for)
        if pres != 0:
            Interest += 1
        line = [search_for, occ]
        result.append(line)
        line_DF[search_for]=pres
    DF.loc[len(DF)] = line_DF
    return result,Interest

def inference(images_dir):
    pertinence_list = []
    b=0
    for file in os.listdir(images_dir):
        if file.endswith(".jpg"):
            image_name = file[:-4]
            # Reading the image
            image = cv2.imread(images_dir+'/'+file)
            image = image[..., ::-1]
            # Convert image to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            threshold_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            # Passe l'image par pytesseract (OCR)
            text = pytesseract.image_to_string(threshold_img) # OCR
            src_path = images_dir+'/'+image_name+'.jpg'
            dst_path_R = "/Users/tchagoue/Documents/AMETHYST/Datas/Dada/PDF/Relevant_image/" + image_name + '.jpg'
            result, interest = pertinence(text,image_name)
            a=0
            if interest != 0:
                a=1
                b+=1
                shutil.copy(src_path, dst_path_R)
            pertinence_list_line=[image_name,a]
            print(pertinence_list_line)
            pertinence_list.append(pertinence_list_line)
    print(pertinence_list)
    print((b/len(pertinence_list))*100) # Percentage of relevant images.
    print(b)
inference('/Users/tchagoue/Documents/AMETHYST/Datas/PDF/Tables_Images')