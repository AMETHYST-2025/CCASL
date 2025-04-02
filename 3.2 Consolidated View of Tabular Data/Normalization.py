# **********************************************************************************************************************
# It is worth mentioning that this program deals with the normalization of a large number of attributes and tables layout.
# It was developed as part of the exploratory approach to the work, and we focused on the most frequent attributes.
# It can serve as a foundation for any work related to attribute functional dependencies, particularly in materials science.
# **********************************************************************************************************************
import numpy as np
import pandas as pd
import zipfile
import os
import csv
import cv2
from numpy import nan
import re
import nltk
from nltk.tokenize import word_tokenize
nltk.download('stopwords')
from nltk.corpus import stopwords
import string
import imageio.v2 as imageio
from string import digits
from nltk.corpus import wordnet as wn
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
from itertools import groupby
import math
from ChemDataExtractor import Abbreviation, EP_signification_detection # ChemDataExtractor can be used to detect chemical abbreviation
from Samples_Normalization import samples_normalization
import jaro
import easyocr
from collections import Counter
from Levenshtein import ratio
# **********************************************************************************************************************
dir_path =  "/Users/tchagoue/Documents/AMETHYST/CCASL/3.2 Consolidated View of Tabular Data/Data/2.2- Sample_Tables_CSV_(after AWS)"
list_pdf = []
numeric_const_pattern = '[-+]? (?: (?: \d* \. \d+ ) | (?: \d+ \.? ) )(?: [Ee] [+-]? \d+ ) ?'
rx = re.compile(numeric_const_pattern, re.VERBOSE)
df_ab = pd.DataFrame(columns=['Abbreviations','Definitions','References','Natures'])
Dict_PP_key = ['LOI (%)', 'Tg (°C)', 'PHRR (kW/m²)', 'THR (MJ/m²)', 'TTI (s)', 'Tensile strength (MPa)',
               'Tonset (C)', 'Elongation at break (%)', 'EHC (MJ/kg)', 'Tmax (C)', 'TSR (m²/m²)', 'TSP (m²)',
               'UL-94', 'Strengh (MPa)',  'Flexural strength (MPa)', 'Strengh (KJ/m²)', 'Tensile modulus (GPa)',
               'Time PHRR (s)', "Young's modulus (MPa)"]

DF_Nom_Column = pd.read_excel("/Users/tchagoue/Documents/AMETHYST/Springer_paper/3.2 Consolidated View of Tabular Data/Data/RNN_Prediction/Norm_column_name.xlsx", index_col=0)
DF_Nom_Column = DF_Nom_Column[['Avant_Norm','Apres_Norm']][:]
# DF_Nom_Column contains the column names that have already been normalized and manually verified. Since we are confident in their results,
# we use them to refine the next results, and so on, in a loop.
print(DF_Nom_Column)
# **********************************************************************************************************************
def Abbreviation_Sigle_detection(df,name,df_ab):
    for dir in list_pdf :
        ch = dir + name + '.pdf'
        if os.path.isfile(ch) :
            EP_signification_detection(ch)
            S,D,N = Abbreviation(ch)
            for j in range(len(S)):
                new_row = [S[j],D[j],name, N[j]]
                df_ab.loc[len(df_ab)] = new_row
            df_ab.to_excel('/Users/tchagoue/Documents/AMETHYST/Code_git/AMETHYST/Normalization/Abbreviations_Lists.xlsx', index=True)
            break
    return  df_ab
    # ******************************************************************************************************************

def token_word(text):
    punctuation = ["(", ")", ";", ":", "[", "]", ",", "'", "-", "/", "."]  # mot a ne pas considerer
    stop_words = stopwords.words('english')
    tokens = word_tokenize(text)
    keywords = [word for word in tokens if not word in stop_words and not word in punctuation]
    return keywords
    # ******************************************************************************************************************

def colonnes_to_skip(df):
    # Detect columns that contain the most non-float strings,
    # we can consider them as descriptive, similar to "Sample".
    column_to_skip = []
    for i in range(len(df.columns)):
        len_str = 0
        len_float = 0
        if df.columns[i]!='Vectors':
            for column_ele in df[df.columns[i]]:
                column_ele = str(column_ele)
                column_ele = column_ele.replace(' ','')
                column_ele = column_ele.replace('+', '')
                column_ele = column_ele.replace('-', '')
                len_str += len(column_ele)
                float_ = rx.findall(column_ele)
                for ii in range(len(float_)):
                    len_float += len(float_[ii])
            if len_str != 0:
                if len_float / len_str < 0.5: # Do not process columns that contain more floats than strings, as they are likely descriptive.
                    if df.columns[i] not in column_to_skip : column_to_skip.append(df.columns[i])
    if 'Vectors' not in column_to_skip : column_to_skip.append('Vectors')
    if 'Samples' not in column_to_skip : column_to_skip.append('Samples')
    return column_to_skip
    # ******************************************************************************************************************

def delete(value,ele_list):
    for ele in ele_list:
        if value.find(ele) != -1:
            value = value.replace(ele, "")
    return value
    # ******************************************************************************************************************

def if_unit(value):
    value = delete(value,[' ','.1','.2','.3','.4','.5','.6','.7'])
    dict_open =[]
    dict_close = []
    if len(value)>2:
        for j in range(len(value)):
            if value[j].find('(') != -1: dict_open.append(j)
            if value[j].find(')') != -1: dict_close.append(j)
        if len(dict_close)*len(dict_open)==1 and len(value)-(dict_close[0]-dict_open[0]+1) == 0: # Verify that the length of the value matches that of the unit.
            return 1
        else:
            if len(dict_close)*len(dict_open)>=1:
                return 2
            else: return 0
    else:
        return 0
    # ******************************************************************************************************************

def jaro_distance(value, list, rate):
    for ele in list :
        #metric = jaro.jaro_metric(ele,value)
        # rate = nlp("Rated flow rate [m³/s]").similarity(nlp("Nominal flow rate [m³/s]"))
        # nlp = spacy.load("en_core_web_sm")
        metric = ratio(ele,value) # Levenshtein distance
        #nlp("Rated flow rate [m³/s]").similarity(nlp("Nominal flow rate [m³/s]"))
        if metric >= rate :
            return [metric, ele]
    return [-1, value]
    # ******************************************************************************************************************

def merge_header(df,image_name):
    empty_H1 = ['',' ','.1','.2','.3','.4','.5','.6','.7','.8','.9']
    cas_particulier = ['Vmax ( (%/°C)','Char at 700 yield °C (%)']
    reader = easyocr.Reader(['en'])
    l_chiffre = 0; temoin = 0
    l_total = 0; presence_vide = 0
    fusion = []
    # Identify cases where the first row of the dataframe needs to be merged with the header.
    column_to_skip = colonnes_to_skip(df)
    for l1 in range(1, len(df.columns)):
        if df.columns[l1] not in column_to_skip:
            chiffre = rx.findall(str(df.iat[0, l1]))
            l_total = len(str(df.iat[0, l1])) + l_total
            if str(df.columns[l1]) in empty_H1 : presence_vide = 1
            for ii in range(len(chiffre)):
                l_chiffre = len(chiffre[ii]) + l_chiffre
    # If line '0' has half as many digits as characters,
    # then we can consider that it is a degeneration.
    if l_chiffre + l_total !=0: #The case where no column is significant
        if l_chiffre+1 <= (l_total / 2):
            # --------------------------------------------------------------------------------------------
            # Detect columns names to merge horizontally, example : ['Tg','(°C)'] -> ['Tg (°C)']
            # Check real file example : j.ijadhadh.2012.10.006_4_0; complexe : j.eurpolymj.2017.08.038_16_0
            # problem, the header took the title : j.matchemphys.2012.11.060_5_0, j.cej.2019.123830_18_0, j.polymdegradstab.2021.109544_8_0, j.polymdegradstab.2016.02.015_3_0,
            for i in range(1, len(df.columns) - 1):
                col_name = df.columns[i] + ' ' + df.columns[i + 1]
                if (jaro_distance(col_name, DF_Nom_Column['Avant_Norm'], 0.95)[0]>0 or if_unit(df.columns[i + 1]) == 1 or col_name in cas_particulier) and if_unit(df.columns[i]) != 2 and df.columns[i] not in  empty_H1: # condt2 : si if_unit=1, alors la colonne suivante ne contient qu'un nom de colonne, j'ai remplacer la condt2, car il faut juste verifier que le groupe de mot est suffisamment proche d'une valeur de reference, ancienne condition : col_name in DF_Nom_Column['Avant_Norm'], ajout d'une condition 3 : pour empecher, l'algorithme d'analyser le vide et l'element suivant, confère j.eurpolymj.2021.110282_3_0
                    print('verifions ceci: ', col_name)
                    df = df.rename(columns={df.columns[i]: col_name, df.columns[i + 1]: col_name + '$'})
            for i in range(1,len(df.columns)):
                if df.columns[i] in empty_H1 :
                    df = df.rename(columns={df.columns[i]: df.columns[i-1] + '$'})
                    if df.iat[0, i]=='' :
                        df.iat[0,i]=df.iat[0,i-1]
            # contradictory case: j.polymdegradstab.2016.04.005_26_0, j.carbpol.2013.05.062_8_0
            # special case : j.matchemphys.2012.11.060_5_0; j.cej.2019.123830_18_0; j.compositesb.2019.107078_6_0;j.compositesb.2020.108271_6_1, j.eurpolymj.2017.05.026_2_0
            # Good example case, duplication  : j.polymdegradstab.2019.07.004_4_0; j.polymdegradstab.2019.07.004_3_0; j.eurpolymj.2021.110638_3_0
            # Good example case, but counter-intuitive : j.eurpolymj.2019.109304_8_0, j.polymdegradstab.2018.01.024_27_0, j.polymdegradstab.2015.07.015_7_0, "of GO : j.compscitech.2021.108671_5_0"
            # ---------------------------------------------------------------------------------------------

            print('ici')
            # merge the header and the first line
            df.rename(columns=lambda x: str(x) + ' ' + str(df[x][0]), inplace=True)
            df = df.iloc[1:]
            df = df.reset_index(drop=True)
            temoin = 1
            # remove numbers due to multiple empty column names, renamed .1, .2, .3 ...
            for l in range(1,len(df.columns)):
                value=str(df.columns[l])
                value1=delete(value,['.1','.2','.3','.4','.5','.6','.7','.8','.9','$'])
                if value1 not in df.columns : df = df.rename(columns={value:value1})
    return df, temoin, presence_vide
    # ******************************************************************************************************************

def identify_col_with_err(df):
    # Duplicate columns with uncertainties
    dict_err = []; test=0
    dict_col = []
    dict_col_all = []
    already_exist = []
    column_to_skip = colonnes_to_skip(df)
    for l in range(len(df.columns)):
        # --------Rename columns with the same name----------
        if df.columns[l] not in already_exist :
            already_exist.append(df.columns[l])
        else:
            already_exist.append(df.columns[l]+'_N')
            test+=1
        if test !=0: df = df.set_axis(already_exist, axis=1)
        #-------------------------------------------------------------
        if df.columns[l] not in column_to_skip:
            numb = 0;esp=0
            for m in range(len(df)):
                if str(df.iat[m, l]).find('+') != -1:
                    numb += 1
                if str(df.iat[m, l]).find(' ') != -1:
                    esp += 1
            if numb >= 1 and esp != 0:  # numb==len(df)
                name = df.columns[l]
                df[name + '_err'] = df.loc[:, [name]]
                dict_err.append(name + '_err')
                dict_col.append(name)
        dict_col_all.append(df.columns[l])
    return dict_err, dict_col, dict_col_all, df
    # ******************************************************************************************************************

def integrity(df,col_skip):
    #Detect CSVs containing good spreadsheets (numeric values)
    not_float=0; len_a =0; len_1 = 0
    for l in df.columns:
        if l not in col_skip:
            for m in range(len(df)):
                if df[l][m] not in [' ','','/','-']:
                    if type(df[l][m]) is not float:not_float = not_float+1
    for ele in df['Samples']:
        num = rx.findall(ele)
        if len(num) !=0:
            if len(num[0]) == len(ele) : len_a+=1
    for ele in df['Vectors']:
        if len(ele)==1: len_1 += 1
    if not_float <= 0.03*((len(df.columns)-len(col_skip))*len(df)): # less than 20% of the values are qualitative. Old condition: not_float <= 3 and len(df.columns)-len(col_skip)>=2
        decision = "Y"
    else:
        decision = "N"
    if len_a >= len(df) / 2: decision = "N" # eliminate the df which have for more than 50% chemical compounds of the figures (the latter must be treated specially), Ex ref: j.compscitech.2015.06.001_4_0
    if len_1 == len(df) : decision = "N" # eliminate the df which does not offer classy compositions. Exception : j.compscitech.2015.06.001_4_0
    return decision,not_float
    # ******************************************************************************************************************

def select_err(dict_err,df):
    # isoler (recuperer) l'incertitude
    if len(dict_err) != 0:
        for name in dict_err:
            ll = 3
            list_ok = []
            for ele in range(len(df[name])):
                value = df[name][ele]
                if value.find('+') != -1:
                    df[name][ele] = value[(value.find('+')) + 1:] #select the error after the '+' sign
                    ll = len(value[(value.find('+')) + 1:]) # recover the length of the chain of the uncertainty value
                    list_ok.append(ele)
            for ele in range(len(df[name])):
                if ele not in list_ok:
                    value = df[name][ele]
                    df[name][ele] = value[-ll:]
    # ******************************************************************************************************************

def delete_err(dict_col,df):
    # remove uncertainty
    if len(dict_col) != 0:
        for name in dict_col:
            len_err = 3 #length of the default error character string
            list_ok =[]
            for ele in range(len(df[name])):
                value = df[name][ele]
                if value.find('+') != -1:
                    df[name][ele] = value[:(value.find('+'))] # removing characters after the '+' sign
                    len_err = len(value[(value.find('+')) + 1:]) # updating error length
                    list_ok.append(ele)
            for ele in range(len(df[name])):
                if ele not in list_ok:
                    value = df[name][ele]
                    df[name][ele] = value[:-len_err]
    # ******************************************************************************************************************

def convert_to_float(df,name_img):
    # convert strings to "float"
    column_to_skip = colonnes_to_skip(df)
    for name in df.columns:
        if name not in column_to_skip:
            all = 0
            for ele in range(len(df)):
                value = df[name][ele]
                value = str(value).replace(",", ".")
                if len(rx.findall(value)) != 0:
                    if value.find('h') == -1 and  value.find('H') == -1 and value.find('%') == -1:
                        df[name][ele] = abs(float(rx.findall(value)[0]))
                        all+=1
            if all == len(df):
                q3 = df[name].quantile(q=0.90) #ref 0.75
                q1 = df[name].quantile(q=0.10) #ref 0.25
                IQR = (q3 - q1)
                b_inf = q1 - 1.5 * IQR
                b_sup = q3 + 1.5 * IQR
                #spot outliers with the laws of quantity
                for ele in range(len(df)):
                    value = df[name][ele]
                    if b_sup < value or value < b_inf:
                        print('valeur aberrante {}'.format(value))  # outlier
    return column_to_skip
    # ******************************************************************************************************************

def replace(k,df, new_name):
    df.rename(columns={df.columns[k]: new_name}, inplace=True)
    return df
    # ******************************************************************************************************************

def Replace(value,strings,R_values):
    for i in range(len(strings)):
        value=value.replace(strings[i],R_values[i])
    return value
    # ******************************************************************************************************************

def identify_col_sample(df): # to review error in file j.cej.2020.125416
    # Identify the “Sample” column and name it as such
    ok=0;j=0;i=0
    col_skip = colonnes_to_skip(df)
    for kk in range(len(df.columns)):
        name=df.columns[kk]
        #--------------------------------------------------
        if len(col_skip) == 3:
            for ele in col_skip:
                if ele not in ['Samples', 'Vectors']:
                    if ele == name :
                        replace(kk, df, "Samples")
                        ok = 1
                        break
        #---------------------------------------------------
        words=token_word(name)
        for Word in words:
            if Word.lower() in ["formula","sample","samples","composition","material"]:i+=1

        if i!=0 and ok ==0 and name in col_skip:
            replace(kk,df, "Samples")
            ok = 1
            break
    if ok ==0 and name in col_skip:
        for kk in range(len(df.columns)):
            for k in range(len(df)):
                name=df.columns[kk]
                words=token_word(df[name][k])
                for Word in words:
                    if Word.lower() in ["epoxy","dgeba","ep"] : j+=1
            if j!=0:
                replace(kk,df, "Samples")
                ok = 1
                break
    if ok ==0:
        replace(0,df, "Samples")
    print(df.columns)
    return df

# HERE CODE FACTORIZED WITH THE Samples_Normalization.py FILE

def col_norm(df, Name):
    df_columns_ = pd.DataFrame(columns=['References', 'Avant_Norm', 'Apres_Norm'])
    def similarity(value, ref, df, k):
        metric = jaro.jaro_metric(value.lower(), ref.lower())
        if metric >= 0.9 :
            replace(k, df, "{}".format(ref))
            print('check ici')
            print(value)
        return df

    def pp_unit_norm(U, PP, Norm, df,value,k):
        unité_ = 0
        Propriété = 0
        for ele in U:
            if value.find(ele) !=-1: unité_ += 1
        for ele in PP:
            if value.find(ele) !=-1: Propriété += 1
        if unité_ != 0 and Propriété != 0:
            df = replace(k, df, "{}".format(Norm))
        return df
    # ******************************************************************************************************************
    unit_presence_col=0; unit_presence_row=0;pres=0
    for k in range(len(df.columns)):
        before_norm = df.columns[k]
        storage=0; modulus=0; tensile=0; young=0; flexural=0; strength=0; MPa=0; GPa= 0;rubbery=0; unit='';Tg=0 ; Impact =0
        value = df.columns[k]
        Unit_det = [
                    [['(', '(s', ')','s'],['TTI', 'TTi'],'TTI (s)'],
                    [['(kW/', 'kW', 'kW/m²','KW'], ['PHRR', 'p-HRR', 'pHRR','pk-HRR','pk HRR','P-HRR'], 'PHRR (kW/m²)'],
                    [['(MJ/', 'MJ/m', 'MJ/m²)', 'MJ'], ['THR', 'thr', 'ThR'], 'THR (MJ/m²)'],
                    [['m2','m²','m'], ['TSR','tsr','Tsr'],'TSR (m²/m²)'],
                    [['m ²/kg','m²/kg', 'm2/kg'],['SEA', 'Sea', 'sea'], 'SEA (m²/kg)'],
                    [['m', 'kg', 'Kg'], ['ASEA'], 'ASEA (m²/kg)'],
                    [['kg', 'Kg'], ['COY','Coy','coy'], 'COY (kg/kg)'],
                    [['kg', 'Kg'],['CO2Y', 'CO2y','CO₂Y'],'CO₂Y (kg/kg)'],
                    [['MJ', 'kg','Kg'], ['EHC', 'Ehc', 'ehc'], 'EHC (MJ/kg)'], # to check
                    [['(kW/', 'kW', 'kW/m²','(KW/'], ['av-HRR', 'Avg HRR'], 'HRR (kW/m²)'], #revenir ici pour plus de details et une exclusion
                    [['m2','m²','m'], ['TSP'],'TSP (m²)'],
                    [['MLR (g/s)'], ['MLR','AvMLR'], 'MLR (g/s)'], # to check !
                    [['m²/kW', '/kW', 's.m²','m²s'], ['FPI'], 'FPI (m²s/kW)'],
                    [['kW/(m².s', 'm².s', 'm² .s','kW/s'], ['FGI'], 'FGI (kW/m²s)'],
                    [['kW', 'm².s', 'm² .s'], ['FIGRA'], 'FIGRA (kW/m²s)'],
                    [['kW/', 'm².s', 'm² .s'], ['FGR'], 'FGR (kW/m²s)'],
                    [['s'], ['TpHRR', 'Time of pHRR', 'tpHRR', 'TPHRR', 'TTPHRR', 'Time to PHRR','t-PHRR','t-pHRR'], 'Time PHRR (s)'],
                    [['g/s', 'g/', '/s'], ['p-COP', 'PCOP'], 'PCOP (g/s)'],
                    [['g/s', 'g/', '/s'], ['p-CO2P', 'PCO2P','p-CO₂P'], 'PCO₂P (g/s)'],
                    [['s', 'm²', 'm2'], ['PSPR'], 'PSPR (m²/s)'],
                    [['MJ', 'kg', 'Kg'], ['AEHC'], 'AEHC (MJ/kg)'],
                    [['J/g', 'gk'], ['HRC'], 'HRC (J/gk)'],
                    [['o (MPa)'], ['o (MPa)'], 'Sigma (MPa)'],
                    [['g','s'], ['AMLR'], 'AMLR (g/s)'],
                    [[''], ['UL - 94','UL-94'], 'UL-94'],
                    [['C'], ['Tonset'], 'Tonset (°C)'],
                    [['%'], ['LOI'], 'LOI (%)'],
                    [['pHRR','PHRR'], ['time','Time'], 'Time PHRR (s)']
                    ]

        #-------------------------------------------------Norm PPs et Unités--------------------------------------------
        for i in range(len(Unit_det)):
            df = pp_unit_norm(Unit_det[i][0], Unit_det[i][1], Unit_det[i][2], df, value,k)
        value = df.columns[k]
        value = Replace(value, ['wt.%','%wt','[s]','wt %','[%]','wt.-%','Superscript (1)','Superscript(1)','Tg(°)','[°C]','S2.cm2','m-superscript(2)','subscript(2)','in wt%'], ['wt%','wt%','(s)','wt%','(%)','wt%','¹','¹','Tg(°C)','(°C)','Ω.cm²','/m²','₂','(wt%)'])
        if value.find('T')!=-1 and value.find('(C)') !=-1 and len(value)<=8: value = value.replace('(C)','(°C)')
        if value.find('UL-94') != -1 : value = 'UL-94'
        try:
            if value[0]==' ': value = value[1:]
            if value[-1] == ' ': value = value[:-1]
        except:
            gjh=1
        replace(k, df, "{}".format(value))

        #-----------------------------------------------Jaro-Winkler Distance-------------------------------------------
        Dict_Ref = ['Yield stress (MPa)','Intensity (MPa)','Time PHRR (s)','Elongation at break (%)','Tonset (°C)',
                    'UL-94','Crosslink density (kmol/m³)']
        for Ref in Dict_Ref :
            df = similarity(value, Ref,df,k)
        #---------------------------------------------------------------------------------------------------------------

        value = Replace(value,[".","/"],[" "," "])
        keywords = token_word(value)
        for kk in keywords:
            if 'storage' == kk.lower(): storage = 1
            if 'modulus' == kk.lower(): modulus = 1
            if 'tensile' == kk.lower(): tensile = 1
            if 'young' == kk.lower(): young = 1
            if 'flexural' == kk.lower(): flexural = 1
            if 'strength' == kk.lower(): strength = 1
            if 'impact' == kk.lower(): Impact = 1
            if 'mpa' == kk.lower(): unit='MPa'
            if 'gpa' == kk.lower() or 'gp'==kk.lower(): unit='GPa'
            if 'mm2'== kk.lower() or 'mm²'== kk.lower(): unit='N/mm²'
            if 'J' == kk.lower(): unit = 'J'
            if 'kj'== kk.lower(): unit='KJ/m²'
            if 'J cm2' == kk.lower(): unit = 'J/cm²'
            if 'tg' == kk.lower():Tg=1
            if 'rubbery'== kk.lower(): rubbery=1
            #if kk.lower() in ['kw','mj','kg','°c']:pres+=1

        if storage+modulus == 2 or rubbery+modulus==2:
            try:
                Cel = int(rx.findall(value)[-1])
            except:
                Cel = ''
            if storage == 1:label="Storage"
            if rubbery == 1:label = "Rubbery"
            if value.find('+') != -1:replace(k, df, "{} modulus (Tg+{}°C) ({})".format(label,Cel,unit))
            else:
                if Cel=='': replace(k, df, "{} modulus ({})".format(label,unit))
                if Cel !='': replace(k, df, "{} modulus ({}°C) ({})".format(label,Cel,unit))
        if tensile+modulus == 2 or young+modulus == 2 or flexural+modulus == 2:
            if tensile == 1: label = "Tensile"
            if young== 1:label="Young's"
            if flexural== 1:label="Flexural"
            replace(k, df, "{} modulus ({})".format(label, unit))
        if flexural+strength == 2:replace(k, df, "Flexural strength ({})".format(unit))
        if tensile + strength == 2: replace(k, df, "Tensile strength ({})".format(unit))
        if modulus == 1 and tensile+young+flexural+storage+strength+rubbery==0:
            replace(k, df, "Modulus ({})".format(unit)) # revenir ici plutard
        if strength == 1 and Impact == 1  and tensile + young + flexural + storage + modulus == 0:
            replace(k, df,"Impact strength ({})".format(unit))
        if Tg == 1 and len(value) <= 7 and value.find('C')!=-1:replace(k, df,"Tg (°C)")

        if df.columns[k] == 'av-COY (kg/kg)':
            print(DF_Nom_Column.loc[DF_Nom_Column['Avant_Norm'] == df.columns[k], 'Apres_Norm'])
            print('jhfvyuhkvh')
        #-------------------------------Manual Correction------------------------------------------
        #"""
        try :
            df_filter = DF_Nom_Column.loc[DF_Nom_Column['Avant_Norm'] == df.columns[k], 'Apres_Norm']
            if df.columns[k] != str(df_filter.tolist()[0]) :
                print('akjzdb', df.columns[k], str(df_filter.tolist()[0]))
                df.rename(columns={df.columns[k]: str(df_filter.tolist()[0])}, inplace=True)
        except :
            kh=1
        #"""
        #--------------------------------------------------------------------------------------------
        after_norm = df.columns[k]
        new_Row = [Name, before_norm, after_norm]
        df_columns_.loc[len(df_columns_)] = new_Row

    return df, df_columns_
    # ******************************************************************************************************************

def transpose_df(df):
    unit_presence_row = 0
    pres = 0; unit_presence_col = 0
    for k in range(len(df)):
        value = df['Samples'][k]
        keywords = token_word(value)
        for kk in keywords:
            if kk.lower() in ['tensile','young','flexural','storage','strength','modulus','c','tg','mpa','gpa', 's','kw/m²', 'tti']: pres += 1
        if value.find("(")!=-1: unit_presence_row +=1
    if unit_presence_row !=0 and pres !=0:
        print('This tables must be transposed')
        df = df.transpose()
        df = df.reset_index(drop=False)
        df.columns = df.iloc[0]
        df = df.reindex(df.index.drop(0)).reset_index(drop=True)
        df.columns.name = None
        print(df)
    return  df
    # ******************************************************************************************************************

def dict_Column(df):
    text=[]
    for k in range(len(df.columns)):
        if df.columns[k] not in ['Samples','Vectors','Ref','',' ','.1']:
            text.append(df.columns[k])
    return text
    # ******************************************************************************************************************

def develop(zf,image_name,DF_final,DF_columns): # Put a loop for the determination of table-n.csv, for n in [1,2,3,4...]
    #print(image_name)
    with zf.open('table-1.csv') as f:
        csv_reader = csv.reader(f, delimiter=';')
        df2 = pd.read_csv(f)
        df2.rename(columns=lambda x: x[1:], inplace=True)
        p=image_name.find('_')
        select=0; detect_more = 1; merge_nbr = 0
        # *************************************************
        for i in range(len(df2)):
            k = 0
            for j in range(len(df2.columns) - 1):
                if type(df2.iat[i, j]) is str:
                    df2.iat[i, j] = df2.iat[i, j][1:]
                if df2.iat[i, j] is nan:
                    k = k + 1
            if k == len(df2.columns) - 1:  # rlocate the line made of NaN
                print('____________________________________')
                df = df2[:i]  # remove “confidence Scores”
                df = df[df.columns[:-1]]
                f_line = i
                print(df) #basic df to clean
                print('************************************')

        # ******************************************Application of functions*******************************************
        while(detect_more==1):
            df, detect_more, pres_vide = merge_header(df, image_name)    #Detect unmerge anomaly and correct it
            if detect_more == 1 : merge_nbr =1
        df = identify_col_sample(df)                                     #Rename all components columns "Samples"
        df = transpose_df(df)
        df, DF_columns_ = col_norm(df,image_name)                        #Normalizing column names
        #Abbreviation_Sigle_detection(df,image_name[:p],df_ab)            #Detect and manage acronyms and abbreviations
        list_vectors, df =samples_normalization(df)                      #Standardize the coding of epoxy-amines
        dict_err, dict_col, dict_col_all, df = identify_col_with_err(df) #Identify column with uncertainty (+/-)
        select_err(dict_err, df)                                         #Clean the error columns
        delete_err(dict_col, df)                                         #Remove the error from the original column
        col_skip=convert_to_float(df, image_name)                        #Convert values to float and identify outliers
        decision,num_NaN = integrity(df,col_skip)                        #Check the usability of the table
        list_name = dict_Column(df)                                      #Copy all the column names
        # **************************************************************************************************************

        df['Ref']=[image_name for i in range(len(df))]
        print(df)
        #--------------------------------------------------
        Colo = ''
        for u in range(len(df)):
            Colo = df['Samples'][u] + ' ' + Colo
        # --------------------------------------------------

        if decision in ['Y'] : # Save the Normalized CSV
            df.to_csv( '/Users/tchagoue/Documents/AMETHYST/Datas/PDF/Tables_Images/CSV_Normalized/' +image_name+'.csv') #
            DF_columns = pd.concat([DF_columns, DF_columns_], ignore_index=True)
        print(decision,num_NaN)
        print('__________________________________________')
        # *************************************************

    return f_line, list_name, list_vectors, decision, dict_col_all, image_name, Colo, df_ab, DF_columns, DF_final, merge_nbr
    # ******************************************************************************************************************

def plot_frequency(List,label):
    DF_plot_freq = pd.DataFrame(columns=['items', 'Frequences'])
    (Patterns, counts) = np.unique(List, return_counts=True)
    results = {x: y for x, y in zip(Patterns, counts)}
    results = dict(sorted(results.items(), key=lambda x: x[1], reverse=True))
    for ele in results:
        print(results)
        new_row = [ele, results[ele]]
        DF_plot_freq.loc[len(DF_plot_freq)] = new_row
    DF_plot_freq.to_excel('/Users/tchagoue/Documents/AMETHYST/Code_git/AMETHYST/Normalization/img/'+label+'.xlsx')
    print('--------------------------------------------------------------------------------')
    print('Le nombre de patterns distinct : {}'.format(len(results)))
    from itertools import islice
    def take(n, iterable):
        return dict(islice(iterable, n))
    print(len(results))
    results=take(30,results.items())
    print('--------------------------------------------------------------------------------')
    Patterns = list(results.keys())
    values = list(results.values())
    plt.bar(range(len(results)), values)
    plt.xticks(range(len(Patterns)), Patterns, rotation=90)
    plt.savefig('/Users/tchagoue/Documents/AMETHYST/Code_git/AMETHYST/Normalization/img/'+label+'.jpeg')
    plt.show()
    # ******************************************************************************************************************

def inference(images_dir):
    Colonne_Top = ['Ref','Samples','Vectors']
    DF_columns = pd.DataFrame(columns=['References', 'Avant_Norm', 'Apres_Norm'])
    DF_final = pd.DataFrame(columns=Colonne_Top)
    Dois = {}; COLO ={}
    error = 0;ok=0; decision='N'
    f1=0; aa=0; f_line=0; nbr_entete_fusionné = 0
    list_name = []
    list_col_name = ''; list_Colo = ''
    list_vectors_all=[]; dict_colonnes_all =[]
    list_fichier = os.listdir(images_dir)
    list_fichier.sort()
    for file in list_fichier:
        if file.endswith(".zip"):
            try:
                image_name = file[:-4]
                with zipfile.ZipFile(images_dir +'/'+ file) as zf:
                    try:
                        f_line, list_name_i, list_vectors, decision, dict_colonnes, doi, Colo, DF_AB, DF_cols, DF_fin, lkshf =develop(zf, image_name, DF_final,DF_columns)
                        # "dict_colonnes" contains all the column names, and "dict_vetors" all the vectors
                        list_name = list_name + list_name_i
                        DF_final=DF_fin; DF_columns = DF_cols
                    except:
                        print('echec ici')
                if decision=='Y':
                    for ele in list_vectors:
                        if ele not in ['',' ','1','2','0']:
                            list_vectors_all.append(ele)
                    for ele in dict_colonnes: dict_colonnes_all.append(ele)
                    for ele in dict_colonnes :
                        if '_err' not in ele :
                            list_col_name = list_col_name+ele+' '
                    Dois[doi]= dict_colonnes ; COLO[doi]=Colo ; list_Colo += Colo
                    aa+=1
                    f1 = f1 + f_line
                    nbr_entete_fusionné += lkshf
            except ValueError:
                error = error+1 # Aberrant CSV

    print(f1)  # Total number of lines possibly usable
    print(aa)
    print('The number of merged tables :' ,nbr_entete_fusionné)
    DF_cols.to_excel('/Users/tchagoue/Documents/AMETHYST/Code_git/AMETHYST/Normalization/Nomenclatures_Noms_Colonnes_Tg_Storage_modulus.xlsx', index=True)
    plot_frequency(list_vectors_all, 'list_chemistry_components_Tg_Storage_modulus')
    A=list(Counter(dict_colonnes_all).values())
    B=list(Counter(dict_colonnes_all).keys())
    PPP = {x: y for x, y in zip(B, A)}
    PPPP = dict(sorted(PPP.items(), key=lambda x: x[1], reverse=True))
    print(PPPP)
    Max_20 =[]
    for e in PPPP:
        Max_20.append(e)
    print(Max_20[:20]) # The 20 most frequent attributes in our data
    #Max_20 = ['LOI (%)', 'Tg(°C)', 'PHRR (kW/m²)', 'THR (MJ/m²)', 'TTI (s)', 'Tensile strength (MPa)', 'Tonset (C)',
    # 'Elongation at break (%)', 'EHC (MJ/kg)', 'Tmax (C)', 'TSR (m²/m²)', 'TSP (m²)', 'DGEBA (g)', 'T5% (C)', 'UL-94',
    # ' Rating', 'UL-94 rating', ' P', ' Dripping', 'Char yield (%)', 'Tp (C)', 'P content (wt%)', 'Test method', '.2',
    # 'UL-94 (3.2 mm)', 'DDM (g)', 'Entry', 'Strengh (MPa)', 'Element C', 'Tmax1 (C)', 'n', 'Filler content',
    # 'Thermal conductivity (W/mK)', 'Complexity of processing', 'Year & References', 'Dripping', 'Residue (wt%)',
    # 'Flexural strength (MPa)', 'Strengh (KJ/m²)', 'Tensile modulus (GPa)', ' o', 'Time PHRR (s)', "Young's modulus (MPa)",
    # 'AH (J/g)', 'Reduction PHRR', '(%) THR', 'DDS (g)', 'Td5% (C)', 'Curing agent', 'Tmax2 (C)', 'PSPR (m²/s)', 'Flexural modulus (GPa)']

    #******************************************** Abbreviation Analysis ************************************************
    #DF_AB = DF_AB.sort_values(by=['Abbreviations'], ignore_index=True)
    #print(DF_AB)
    #DF_AB.to_excel('Abbreviations_List.xlsx', index=True)
    #*******************************************************************************************************************

    return list_col_name, list_Colo, Dois

list_col_name, list_Colo, Dois = inference(dir_path)


