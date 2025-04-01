#----------------------------------------------------Example Image Preprocessing------------------------------------------------
# Add borders to your images to improve processing in AWSTextract
import cv2
from skimage import data
import matplotlib.pyplot as plt
from skimage.util import img_as_ubyte
from skimage import io
from PIL import Image, ImageOps
import os

filename='/Users/tchagoue/Documents/AMETHYST/Datas/Dataset_IMG_1_Rule-based_classification/R/j.polymdegradstab.2019.06.020_6_0.jpg'
label = 'bordure_black'
def border(img_path, file):
    add = 300
    filename = img_path + '/' + file
    im = cv2.imread(filename)
    shape = im.shape
    if shape[0] < 100 or shape[1] < 100: add = 400
    img = Image.open(filename)
    img_with_border = ImageOps.expand(img, border=add, fill='black')  # white
    img_with_border.save('/Users/tchagoue/Documents/AMETHYST/Datas/PDF/Tables_Images/R_AWS/' + file)

path = "/Users/tchagoue/Documents/AMETHYST/Datas/PDF/Tables_Images/R"
for file in os.listdir(path):
        border(path, file)

#-----------------------------------------------------Textract API------------------------------------------------------
# connect to the Amazon Textract API
from textractor import Textractor
from textractor.data.constants import TextractFeatures
import boto3

aws_access_key_id = '**************' # create your own
aws_secret_access_key = '**************' # create your own
client = boto3.client('textract', region_name='eu-west-3', aws_access_key_id = 'AKIAYI7XPQ5OU6PBAFNA', aws_secret_access_key= 'FEcNGoxKCtFzIK6F0uF++7jmiOZmrK6qjXGOokcF')
with open('C:/Users/tchag/Desktop/AMETHYST/Image_interest_classifier/0014-3057(87)90125-x_2_0.jpg', 'rb') as file: #example
    img_test = file.read()
    bytes_test = bytearray(img_test)
response = client.analyze_document(Document={'Bytes': bytes_test},FeatureTypes = ['TABLES'])
print(response)
blocks=response['Blocks']
print(blocks)
blocks_map = {}
table_blocks = []
for block in blocks:
    blocks_map[block['Id']] = block
    if block['BlockType'] == "TABLE":
        table_blocks.append(block)
"""
if len(table_blocks) <= 0:
    return "<b> NO Table FOUND </b>"
"""
csv = ''
for index, table in enumerate(table_blocks):
    csv += generate_table_csv(table, blocks_map, index +1)
    csv += '\n\n'
output_file = 'Convert_CSV.csv'
# replace content
with open(output_file, "wt") as fout:
    fout.write(csv)
# show the results
print('CSV OUTPUT FILE: ', output_file)
# Saves the table in an excel document for further processing
#document.tables[0].to_excel("output.xlsx")