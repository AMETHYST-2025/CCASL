#pip3.9 install Pillow==8.2.0 #brew install poppler
import os
from pdf2image import convert_from_path
import shutil
import cv2
import numpy as np
import os
import layoutparser as lp
from matplotlib import pyplot as plt
from tqdm import tqdm
from time import sleep
import pytesseract
import pandas as pd
import requests
import PIL
from PIL import Image
np.random.seed(42)
#model = lp.Detectron2LayoutModel('lp://HJDataset/faster_rcnn_R_50_FPN_3x/config')
model = lp.Detectron2LayoutModel('lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config',extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.81],
                                 label_map={0: "Text", 1: "Title", 2: "List", 3:"Table", 4:"Figure"})

dir_pdf = '/Users/tchagoue/Documents/AMETHYST/Datas/PDF/'

def plt_imshow(title, image):
    # convert the image frame BGR to RGB color space and display it
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	plt.imshow(image)
	plt.title(title)
	plt.grid(False)
	plt.show()

def save_detections(table_blocks, image, image_name, titre , dirrr):
    #print(range(len(table_blocks)))
    save_dir = dirrr + 'Tables_Images/'
    for j in range(len(table_blocks)):
        x_1, y_1, x_2, y_2 = table_blocks[j].block.x_1, table_blocks[j].block.y_1, table_blocks[j].block.x_2, table_blocks[j].block.y_2
        cropped = image[int(y_1):int(y_2), int(x_1):int(x_2)] # Bounding box
        file_name = image_name+'_'+str(j)+'.jpg'
        print(save_dir+file_name)
        cv2.imwrite(save_dir+file_name, cropped)
        status = cv2.imwrite(save_dir+file_name, cropped)
        if status:
            print("Saved ", file_name)
    

def inference(images_dir, dirrr):
    table_blocks_list = []
    # Getting images from the directory
    for file in os.listdir(images_dir):
        if file.endswith(".jpg"):
            image_name = file[:-4]  
            # Reading the image
            image = cv2.imread(images_dir+'/'+file)
            image = image[..., ::-1]
            # Running Inference
            layout = model.detect(image)
            # Extracting Tables
            table_blocks = lp.Layout([b for b in layout if b.type=="Table"])
            figure_blocks = lp.Layout([b for b in layout if b.type=='Figure'])
            #figure_title = lp.Layout([b for b in layout if b.type == 'Text'])
            #print(figure_title)
            figure_title = 'titre'

            table_blocks = lp.Layout([b for b in table_blocks \
                   if not any(b.is_in(b_fig) for b_fig in figure_blocks)])
            h, w = image.shape[:2]

            left_interval = lp.Interval(0, w/2*1.05, axis='x').put_on_canvas(image)

            left_blocks = table_blocks.filter_by(left_interval, center=True)
            left_blocks.sort(key = lambda b:b.coordinates[1])

            right_blocks = [b for b in table_blocks if b not in left_blocks]
            right_blocks.sort(key = lambda b:b.coordinates[1])

            # And finally combine the two list and add the index
            # according to the order
            table_blocks = lp.Layout([b.set(id = idx) for idx, b in enumerate(left_blocks + right_blocks)])
            save_detections(table_blocks, image, image_name, figure_title, dirrr)
            #table_blocks_list.append(table_blocks)
    print("Analyse terminée pour ce PDF")
    #fin()
    #convert_csv('IMG_CSV_/')
    #return table_blocks_list


def pdf_inference(pdfName, file_name, dirrr):
    # Converting each page to an image
    path = os.getcwd()
    PDF_file = pdfName
    if os.path.exists(path+'/pdf_images_new'):
        shutil.rmtree(path+'/pdf_images_new')
    os.mkdir(path+'/pdf_images_new')

    pages = convert_from_path(PDF_file) #, dpi=100, grayscale=True
    image_counter = 1

    for page in pages: 
        filename = file_name+'_'+str(image_counter)+".jpg" # we can track down the DOI by increasing it in the image name
        # st.write(filename)
        filepath = path+"/pdf_images_new/" + filename

        page.save(f'{filepath}', 'JPEG') 
        image_counter = image_counter + 1

    filelimit = image_counter-1
    # Running inference on the images
    inference(path+'/pdf_images_new', dirrr)

def Run_all(dir_all):
    for file in os.listdir(dir_all):
        if file.endswith(".pdf"):
            try:
                file_name = file[:-4]
                dirrr = dir_all
                pdf_inference(dir_all+'/'+file, file_name, dirrr)
            except:
                print('failed')


Run_all(dir_pdf)

#----------------------------------------------Dict of PDF that has been process-------------------------------------------

