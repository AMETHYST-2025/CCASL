from chemdataextractor import Document
from chemdataextractor.model import Compound
from chemdataextractor.doc import Paragraph, Heading
from chemdataextractor.model import BaseModel, StringType, ListType, ModelType
import re
from chemdataextractor.parse import R, I, W, Optional, merge
from chemdataextractor.parse.base import BaseParser
from chemdataextractor.utils import first
from PyPDF2 import PdfReader
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
from nltk.tokenize import word_tokenize

numeric_const_pattern = '[-+]? (?: (?: \d* \. \d+ ) | (?: \d+ \.? ) )(?: [Ee] [+-]? \d+ ) ?'
rx = re.compile(numeric_const_pattern, re.VERBOSE)
#---------------------------------Abbreviation Detection-------------------------
def Abbreviation(pdf):
    S=[]; D=[]; N=[]
    # Lire les pages du pdf source
    reader = PdfReader(pdf)
    text = ''
    for p in range(len(reader.pages)):
        page = reader.pages[p]
        text = text + page.extract_text()
    p = Paragraph('u"'+text+'"')
    T = (p.abbreviation_definitions)
    def merge(M):
        Mf = ""
        for ele in M:
            Mf = Mf+" "+ele
        return Mf[1:]
    for i in range(len(T)):
        s = T[i][0][0]
        d = merge(T[i][1])
        n = T[i][2]
        S.append(s); D.append(d.replace(';','')); N.append(n)
    return S, D, N

#---------------------------------EP(Epoxy - amine) Detection-------------------------

def EP_signification_detection(pdf):
    EPs = []
    duplicate = []
    # Lire les pages du pdf source
    reader = PdfReader(pdf)
    text = ''
    for p_nb in range(len(reader.pages)):
        page = reader.pages[p_nb]
        text = text + page.extract_text()
    text = re.sub(u'[^\u0020-\uD7FF\u0009\u000A\u000D\uE000-\uFFFD\U00010000-\U0010FFFF]+', '', text)
    p = Paragraph('u"' + text + '"')
    #-------------------------------------------------------------------------------------------------
    key_word_gold = ['Experimental and method','Experimental method', 'Materials and method',
                     'Experimental part','Experimental Part','Experimental section','Experimental Materials',
                     'Experimental procedure','Materials and Method','Experimental Section']

    key_word = ['Experimental','Material','EXPERIMENTAL','MATERIAL','Method']
    #------------------Probability of presence in the title of "Material and Method"----------------------------
    # 1 point on les index_find by presence key_word
    # 2 point if num + key_word
    # 2 point if key_word_gold
    # 3 point if num + key_word_gold
    # Select the section of text indexes which will have the greatest number of attention (coefficient)
    probability = []; G=[]; ok = 0
    for ele in key_word :
        start = 0
        Idx = text[start:].find(ele)
        already = []
        while Idx !=-1 and Idx not in already:
            # Taking into account the presence of title levels in figures, example : "2. Experimental mothod" ou "2.1 Materials"
            poids = 1
            num = rx.findall(text[Idx-6:Idx])
            if len(num) !=0:
                poids = 2
                if ele == 'Method': poids=1.5
                if num[0] in ['2.', '2.1', '2.2'] : poids =3
            vec = [Idx, poids, ele]
            probability.append(vec)
            start = Idx + len(ele)
            already.append(Idx)
            Idx = text[start:].find(ele)
    for ele in key_word_gold :
        Idx = text.find(ele)
        if Idx !=-1 :
            vec = [Idx, 3, ele]
            probability.append(vec)

    for ele in probability :
        if ele[1] == 3: G.append(ele[0])          # Priority 3 for full titles 'key_word_gold' and 'numero+key_word'
    if len(G) == 0 :
        for ele in probability :
            if ele[1] == 2 : G.append(ele[0])     # Then level 2
    if len(G) == 0:
        for ele in probability :
            if ele[1] == 1.5 : G.append(ele[0])
    if len(G) == 0:
        for ele in probability :
            if ele[1] == 1 and ele[0]>1000: G.append(ele[0])  # Consider simple elements in the last position, take from the 1000th character to avoid keywords in the title (because in the first 1000 characters we very often have the Summary and/or the material and method and/or the introduction)

    start_material = 0 # defaut (take the resume)
    if len(G) != 0 : start_material = min(G)      # Comment
    print('-----------------------------------------------')
    print(probability)
    Material = text[start_material:start_material+1000]
    print(Material)
    print('-----------------------------------------------')

    T = p.cems
    for ele in T:
        sub_mat = [ele.text, ele.start-2, ele.end-2]
        EPs.append(sub_mat)
    # We are interested in the chemical compounds present in the portion determined to be material.
    EPs_Materials = [ele for ele in EPs if ele[:][1] > start_material and ele[:][1] < start_material + 1000]
    print(EPs_Materials)

    return EPs_Materials, Material
    #*******************************************************************************************************************

#Abbreviation('/home/tchagoue/Téléchargements/10.1016@j.compositesb.2019.05.015.pdf')





