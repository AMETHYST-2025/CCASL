#*********************************************************************************************************************************************************#
import requests
from crossref.restful import Works
from scidownl import scihub_download
import random
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from time import sleep
import os
import pandas as pd
from datetime import date
from io import StringIO
import sys
today = date.today()
#**************************************# Redo this experiment with all the journal titles and corresponding prefixes you can find****************************************#
save_file_name = '/Users/tchagoue/Documents/AMETHYST/Code_git/AMETHYST/Download_DOI/DOI_TG_Storage_Journals.txt'
path_history = "/Users/tchagoue/Documents/AMETHYST/Datas/Download_Tg_Storage_Modulus_Journals/Downloaded_Files.xlsx"
journal_title = 'Journal of Applied Polymer Science'
prefix = '10.1002'

"""
Some examples provided in the paper :
    -Prefixe : 10.1016
    -Corresponding Journal :  Composites Part B : Engineering  , Chemical Engineering Journal , Progress in Polymer Science , European Polymer Journal , Carbohydrate Polymers , Composites Science and Technology , Polymer Degradation and Stability , Materials Chemistry and Physics
Prefixe-Journals: Polymers-10.3390, Polymers Journal-10.1038, American Chemical Society-10.1021
"""
#*********************************************************************************************************************************************************#
def path_to_pdf(journal_title):
    PATH = "/Users/tchagoue/Documents/AMETHYST/Datas/Download_Tg_SM_Journals"+"/"+journal_title
    isExist = os.path.exists(PATH)
    if not isExist :
        os.makedirs(PATH)
    return PATH

#*********************************************************************************************************************************************************#
def to_lower(text):
    return ' '.join([w.lower() for w in word_tokenize(text)])
#......................................................................................
def searchInName(Title, key):
    occurrences = 0
    presence =0
    #Title = to_lower(Title)
    tokens = word_tokenize(Title)

    punctuation = ["(",")",";",":","[","]",",","'","-","/"]
    stop_words = stopwords.words('english')
    keywords = [word for word in tokens if not word in stop_words and  not word in punctuation]
    for k in keywords:
        if key == k.lower(): occurrences+=1 # lower is converting  "Hello" to  "hello" or "HELLO" to "hello"
    if occurrences != 0:
        presence = 1; # # If the searched word exists in the text, "Presence" is set to 1; otherwise, it is set to 0.
    return occurrences, presence

#-------------------------------------------- File already downloaded -------------------------------
"""
# Execute only once to create the file 'Downloaded_Files.xlsx'
import pandas as pd
DF_DOI = pd.DataFrame(columns=['DOI','Journal_Title','Prefix','Download_on', 'KeyW_select'])
DF_DOI.to_excel(path_history, index=True)
"""

occ= 0
pres= 0
key_words = ['epoxy','amine','resin','storage modulus','glass transition temperature']
#---------------------------------------------------------------------------------------------------

# create the works object
works = Works()
w1 = works.query()

# Filter the results from SCI-hub Data base, to have all publications from journal
w1 = w1.filter(prefix=prefix, container_title=journal_title)
print("Total results: %s" % str(w1.count()))

#*********************************************************************************************************************************************************#
def find_ind_list(doi,list):
    present = 0
    for ele in list:
        if doi == ele or prefix+'/'+doi == ele :
            present += 1
    return present

DF_DOI = pd.read_excel(path_history, index_col=0)
with open(save_file_name, 'w') as f:
    for idx, item in enumerate(w1): #
        if idx % 10 == 0: print(idx)
        if idx % int(10*random.uniform(0,1)+1) == 0: sleep(random.uniform(0,1))
        if idx >=4000: #
            #if idx >11847 : # # To analyze the remaining files if the program gets disconnected along the way
            result = []; Telecharger = 0
            date = item['published']['date-parts'][0][0]
            doi = item['DOI']
            if idx % int(10 * random.uniform(0, 1) + 1) == 0: sleep(random.uniform(0, 1))
            #sleep(random.uniform(0,0.3))
            #if date >= 2014 : # Do not download publications older than 10 years ?

            try :
                title = str(item['title'])
                #print(idx,doi, title)
                Interest = 0; KeyW_select = ''
                for search_for in key_words :
                    occ, pres = searchInName(title,search_for)
                    if pres !=0: Telecharger = 1
                    line = [search_for,pres]
                    result.append(line)

                if Telecharger ==1 and find_ind_list(doi,DF_DOI['DOI']) == 0:
                    print(idx,result)
                    f.write('https://doi.org/' + doi + '\n')
                    # DOWNLOAD THE DOI PDF :
                    paper = ('https://doi.org/' + doi)
                    paper_type = "doi"
                    #out = ('./download_from_dois_new/'+ doi + '.pdf')
                    PATH = path_to_pdf(journal_title)
                    out = (PATH+'/' + doi[8:] + '.pdf')

                    tmp = sys.stdout
                    my_result = StringIO()
                    sys.stdout = my_result
                    scihub_download(paper, paper_type=paper_type, out=out)
                    sys.stdout = tmp
                    h = my_result.getvalue()
                    print('-----------------------------------')
                    print(h)
                    print('-----------------------------------')
                    if '100%' in h :
                        new_row = ({'DOI': doi,'Journal_Title' : journal_title, 'Prefix' : prefix, 'Download_on' : today, 'KeyW_select' : KeyW_select})
                        DF_DOI.loc[len(DF_DOI)] = new_row
                        DF_DOI.to_excel(path_history, index=True)
                        print('ok')
            except:
                None

""" The file 'Downloaded_Files' and the downloaded PDF documents are located in : /CCASL/3.1 PDF Document Acquisition/Data_Samples/Approach-a """
#*********************************************************************************************************************************************************#
