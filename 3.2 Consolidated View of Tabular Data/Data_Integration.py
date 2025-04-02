#***********************************************************************************************************************
# https://www.cs.cornell.edu/johannes/papers/2002/kdd2002-spam.pdf
# https://www.philippe-fournier-viger.com/spmf/CM-SPAM.php
import numpy as np
import pandas as pd
import zipfile
import os
import csv
from numpy import nan
import re
from Levenshtein import ratio
import nltk
from nltk.tokenize import word_tokenize
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
import jaro
from collections import Counter
from itertools import combinations
#***********************************************************************************************************************
                                               #Step-1: Integration by part#
#***********************************************************************************************************************
# In this first part, it is a question of merging all the normalized csv using the "ids of the PDF and Sample Names as key"
# , in other words, CSVs from the same family (i.e. the same PDF) will be  merged together, and therefore
# the data which also has the same sample will be combined. After this merger by part, all the sub-CSV_family obtained
# will be merged together by the superposition of identical attributes.
#***********************************************************************************************************************

label = 'family_integration'
csv_path = '/Users/tchagoue/Documents/AMETHYST/Code_git/AMETHYST/Tables_Images/CSV_Normalized' # folder of all normalized CSVs
family_path = '/Users/tchagoue/Documents/AMETHYST/Code_git/AMETHYST/Tables_Images/CSV_Normalized/family' # folder of CSVs merge by family

def in_same_file(csv_path): # allows you to group the CSVs from the same pdf by their identifier.
    list_csv = os.listdir(csv_path)
    cluster = []
    while len(list_csv) !=0:
        for ele_i in list_csv :
            clus_i = []
            clus_i.append(ele_i)
            try:
                list_csv.remove(ele_i)
            except :
                None
            for ele in list_csv :
                if ele[:ele.find('_')]==ele_i[:ele_i.find('_')]:
                    clus_i.append(ele)
                    list_csv.remove(ele)
            if len(clus_i)>=2 : cluster.append(clus_i)
    print(cluster)
    print("Total number of csv : ", len(os.listdir(csv_path)))
    print("Number of PDFs providing more than one table (or Nbr pairs): ",len(cluster))
    total =0
    for doc in cluster:
        total+=len(doc)
    print('Number of non-isolated tables: ',total)
    return  cluster
    # ******************************************************************************************************************
def unique_attribut(df1, df2):
    doublon = 0
    for ele in df1.columns :
        if ele in df2.columns and ele not in ['Samples']:
            doublon +=1
    if doublon == 0 :
        return True
    else :
        return False
    # ******************************************************************************************************************

def similarity(vec1,vec2):
    count=0
    for i,j in zip(vec1,vec2):
        if i==j or (str(i)== 'nan'  or str(j)=='nan') :
            count+=1
    rate = int(100*count/len(vec1))
    #print(rate)
    if rate==100 :
        decision = 1
    else:
        decision = 0
    return decision, rate
    # ******************************************************************************************************************

def suppression_doublons(DF_final):
    # --------------------------------------------Detect lines to merge-------------------------------------------------
    # j.cej.2012.01.134 detect if it is an ocr error
    redo = 0
    fusion_list = []
    exclude = []
    for i in range(len(DF_final)):
        if i not in exclude:
            list_sim = []
            list_sim.append(i)
            exclude.append(i)
            for j in range(len(DF_final)):
                if j not in exclude:
                    decision, rate = similarity(list(DF_final.iloc[i])[1:], list(DF_final.iloc[j])[1:])
                    #[1:] so as not to compare the "Samples" values, because they are not necessarily normalized
                    if decision == 1:
                        list_sim.append(j)
                        exclude.append(j)
            if len(list_sim) > 1:
                fusion_list.append(list_sim)
    print(fusion_list)
    # ----------------------------------merge the detected lines------------------------------
    if len(fusion_list) != 0:
        for vec in fusion_list:
            new_vec = []
            if len(vec) >= 3:
                vec = vec[:2] # this allows this case to be partially processed, in the condition len(vec)==2, the duplicate deletion function will be restarted once again until len(vec)<4
                redo = 1
            if len(vec) == 2:
                for i, j in zip(list(DF_final.iloc[vec[0]]), list(DF_final.iloc[vec[1]])): #[1:] so as not to compare the "Samples" values, because they are not necessarily normalized
                    if str(i) == str(j):
                        new_vec.append(i)
                    else:
                        if str(i) == 'nan':
                            new_vec.append(j)
                        else:
                            new_vec.append(i)
                            if str(j)!='nan':
                                print(i,'  ;  ',j)
            DF_final.loc[len(DF_final)] = new_vec
        # Removing duplicates :
        for VEC in fusion_list:
            for idx in VEC:
                DF_final = DF_final.drop(index=idx)
    DF_final = DF_final.reset_index(drop=True)
    return DF_final, redo

def integrate_family(disk,path):
    nbr_vide = 0
    def drop_empty(df):
        df.dropna(subset=['Samples'], inplace=True)
        return df
    for family in disk:
        DF_family = pd.DataFrame(columns=['Samples'])
        vectors = {}
        ref = family[0][:family[0].find('_')]
        for file in family:
            if file[-4:] == '.csv':
                csv_n = pd.read_csv(path+'/'+file, index_col=0)
                csv_n = drop_empty(csv_n)
                #csv_n['Samples'] = csv_n['Samples'].replace(np.nan, '')
                csv_n['Samples'] = csv_n['Samples'].astype(str)
                for S,V in zip(csv_n['Samples'],csv_n['Vectors']):
                    vectors[S]=V
                col = list(csv_n.columns); col.remove('Vectors') ; col.remove('Ref')
                csv_n=csv_n[col][:]
                #print(csv_n,file)
                if unique_attribut(csv_n, DF_family)==True :
                    DF_family = DF_family.merge(csv_n, on="Samples", how='outer')
                else:
                    DF_family = pd.concat([csv_n, DF_family], ignore_index=True)
                DF_family = DF_family.reset_index(drop=True)
        DF_family['Vectors'] = [vectors[S] for S in DF_family['Samples']]
        DF_family['Ref'] = [ref for i in range(len(DF_family))]
        DF_family = DF_family.drop_duplicates()
        DF_family = DF_family[['Samples'] + [x for x in DF_family.columns if x != 'Samples']] # To ensure that 'Samples' is the first columns.
        DF_family = DF_family.reset_index(drop=True)
        DF_family, redo = suppression_doublons(DF_family)
        while redo==1:
            DF_family, redo = suppression_doublons(DF_family)
        DF_family.to_csv(family_path + '/' + ref + '.csv') # save DF_family to family_path
    print(DF_family.describe())

def integrate(csv_path, family_path):
    Colonne_Top = ['Ref', 'Samples', 'Vectors']
    DF_final = pd.DataFrame(columns=Colonne_Top)
    count=0
    for file in os.listdir(csv_path):
        if file[-4:] == '.csv' and file[:file.find('_')]+'.csv' not in os.listdir(family_path): # firt,
            csv_n = pd.read_csv(csv_path+'/'+file, index_col=0)
            DF_final = pd.concat([DF_final, csv_n], ignore_index=True)
            DF_final = DF_final.reset_index(drop=True)
            count+=1
    print(count)
    for file_family in os.listdir(family_path) :
        if file_family[-4:] == '.csv':
            csv_n = pd.read_csv(family_path + '/' + file_family, index_col=0)
            DF_final = pd.concat([DF_final, csv_n], ignore_index=True)
            DF_final = DF_final.reset_index(drop=True)
    DF_final = DF_final.drop_duplicates()
    DF_final = DF_final.reset_index(drop=True)
    DF_final.to_csv('/Users/tchagoue/Documents/AMETHYST/Code_git/AMETHYST/Data_Integration/DF_All_'+label+'.csv')
    print(DF_final)
    print(DF_final.describe()) # Total : [2659 rows x 1862 columns]
    return DF_final

Family = in_same_file(csv_path) # Family is a list of family list
integrate_family(Family, csv_path) # Create the family_csv in the family_path, it is empty at the start
DF_final  = integrate(csv_path, family_path) # Dataframe of all CSVs, obtained by integrating the family_csv and the csv which does not have one..


#***********************************************************************************************************************
                                               #Step-2: Extraction of SM_bf, SM_af, Tg#
#***********************************************************************************************************************
# In this part we will extract the data corresponding to the attribute that interests us using the algorithm-3:
# "Integration: creation of a consolidated view".
#***********************************************************************************************************************



numeric_const_pattern = '[-+]? (?: (?: \d* \. \d+ ) | (?: \d+ \.? ) )(?: [Ee] [+-]? \d+ ) ?'
rx = re.compile(numeric_const_pattern, re.VERBOSE)
df = pd.read_csv("'/Users/tchagoue/Documents/AMETHYST/Code_git/AMETHYST/Data_Integration/DF_All_'+label+'.csv'", index_col=0)
df = DF_final
def is_float(value):
    T = rx.findall(str(value))
    if len(T) == 0:
        return  False
    else:
        return True
def Levenshtein(value1, value2):
    result = (ratio(str(value1), str(value2)))*100
    return result

def Tg_Modulus_extraction(excel_file):
    df_Tg_Modulus = pd.DataFrame(columns=['Ref', 'Samples', 'Vectors', 'Tg (°C)','SM_Av_Tg (MPa)','SM (MPa)','SM_Ap_Tg (MPa)', 'Attributs'])
    df = pd.read_csv(excel_file, index_col=0) #read_excel
    def detect_tg(value):
        if 'tg' in value.lower() and 'tga' not in value.lower():
            return True
        else:
            return False
        # **************************************************************************************************************

    def detect_SM(value, tg_val):
        decision= False; tg = None
        keyword = ['storage modulus','tensile modulus']
        keyword_cap = ["E'","G'"] #E (30°C) (Mpa)
        for key in keyword:
            if key in value.lower():
                decision = True
        for key in keyword_cap:
            if key in value:
                decision =True
        if 'E' in value and ('mpa' in value.lower() or 'gpa' in value.lower()):
            print("verifier")
            decision=True
        if decision == True:
            num = rx.findall(str(value))
            if len(num) !=0:
                tg= float(num[0])
                if 'Tg+' in value and tg_val !=None:
                    tg=tg+tg_val
                    print('******************','\n Tg'+num[0],':',tg)
        return decision, tg
        # **************************************************************************************************************

    def select_best_Tg(list):
        best=Levenshtein(list[0],'Tg (°C)'); Tg=list[0]
        for ele in list[1:]:
            rate = Levenshtein(ele,'Tg (°C)')
            if rate > best: Tg=ele
        return Tg
        # **************************************************************************************************************

    def rank(List,state,a):
        rank_componemt = {}
        if len(List)==0:
            return None
        for ele in List:
            rank_componemt[ele[0]] = int(ele[1])
        rank_componemt = dict(sorted(rank_componemt.items(), key=lambda item: item[1], reverse=state))
        list_ = []
        for ele in rank_componemt: list_.append(ele)
        if a==0:
            print(list_)
            return list_[0]
        if a==1:
            return list_
        # **************************************************************************************************************

    def identified_Tg_col(col_list):
        Tg_list = []
        for col in col_list:
            if detect_tg(col) == True:
                Tg_list.append(col)
        if len(Tg_list) >= 2:
            Tg = select_best_Tg(Tg_list) #come back to this function later for different structures
        if len(Tg_list) == 1: Tg = Tg_list[0]
        if len(Tg_list) == 0: Tg = None
        return Tg
        # **************************************************************************************************************

    def identified_SM(col_list,tg_val):
        list_SM=[]
        for col in col_list:
            decision, SM_temp = detect_SM(col, tg_val)
            if decision==True:
                col_meta_tg = [col,SM_temp]
                list_SM.append(col_meta_tg)
        if len(list_SM) !=0: print(list_SM)
        if len(list_SM) == 0:
            return None, None, None # Here we have no value of Storage_Modulus (SM)
        if len(list_SM) == 1:
            print('tg_val:', tg_val)
            if tg_val == None or list_SM[0][1] == None:
                return None, list_SM[0][0], None# If no value of Tg, then not possible to compare, we centralize the value.
            else:
                num=rx.findall(str(list_SM[0][1]))[0]
                if tg_val>float(num): return list_SM[0][0], None, None
                if tg_val==float(num): return None, list_SM[0][0], None
                if tg_val<float(num): return None, None, list_SM[0][0]
        if len(list_SM)>=2:
            list_SM_av_Tg=[]
            list_SM_ap_Tg = []
            SM_Tg = []
            unranked=[]
            if tg_val != None:
                print('tg_val:', tg_val)
                for SM in list_SM:
                    if SM[1] != None:
                        if tg_val < float(SM[1]) or 'befor' in SM[0].lower(): list_SM_ap_Tg.append(SM)
                        if tg_val > float(SM[1]) or 'after' in SM[0].lower(): list_SM_av_Tg.append(SM)
                        if tg_val == float(SM[1]): SM_Tg.append(SM)
                    if SM[1]== None: unranked.append(SM)
                if len(unranked)==0: #pure case
                    return rank(list_SM_av_Tg,False,0), rank(SM_Tg, True,0) ,rank(list_SM_ap_Tg,True,0) # here we are in the best of all worlds.
                if len(list_SM_av_Tg)==0 and len(list_SM_ap_Tg)==0 and len(SM_Tg)==0: #opposite pure case
                    if len(list_SM)==2: return list_SM[0][0], None, list_SM[1][0]
                    if len(list_SM)>=3: return list_SM[0][0], list_SM[int(len(list_SM)/2)][0],list_SM[-1][0]
                if len(unranked) !=0:#hybrid case
                    av = rank(list_SM_av_Tg, False, 0)
                    ap = rank(list_SM_ap_Tg, True, 0)
                    mid = rank(SM_Tg, True, 0)
                    if len(list_SM)==2: return av,unranked[0][0],ap
                    if len(list_SM)==3:
                        if len(unranked)==1: return av,unranked[0][0],ap
                        else:
                            if av != None: return av, unranked[0][0], unranked[1][0]
                            if ap != None: return unranked[0][0], unranked[1][0], ap
                            if mid != None: return unranked[0][0],mid, unranked[1][0]
                    if len(list_SM)>3: # evolve the previous laws
                        if av != None and mid != None and ap != None: return av, mid, ap
                        if av != None and ap != None: return av, unranked[int(len(unranked) / 2)][0], ap
                        if av != None and mid != None: return av, mid, unranked[-1][0]
                        if ap != None and mid != None: return unranked[0][0], mid, ap
                        if av != None: return av, unranked[int(len(unranked)/2)][0], unranked[-1][0] #take the first value (which is certain), and the last as being the one after the Tg
                        if ap != None: return unranked[0][0], unranked[int(len(unranked)/2)][0], ap
                        if mid != None: return unranked[0][0], mid, unranked[-1][0]
            SM_temperature=[]
            unstructured = []
            if tg_val == None:
                for SM in list_SM:
                    if SM[1] != None: SM_temperature.append(SM)
                    if SM[1] == None: unstructured.append(SM)
                rang_SM = rank(SM_temperature, False, 1)
                if len(unstructured)==0: #pure case
                    if len(rang_SM) == 2: return rang_SM[0], None, rang_SM[1] #
                    if len(rang_SM) == 3: return rang_SM[0], rang_SM[1], rang_SM[2]
                    if len(rang_SM) > 3: return rang_SM[0], rang_SM[int(len(rang_SM)/2)], rang_SM[-1]
                if len(SM_temperature) == 0:#inverse pure case (neither tg_val, neither temperature of Storage Modulus)
                    if len(unstructured) == 2: return unstructured[0][0], None, unstructured[1][0]
                    if len(unstructured) >= 3: return unstructured[0][0], unstructured[int(len(unranked)/2)][0], unstructured[-1][0]
                if len(unstructured) !=0 : # hybrid case
                    if len(rang_SM) == 1:
                        if len(unstructured) == 1: return unstructured[0][0], None, rang_SM[0]  #given a total >=2, we will prioritize the Storage modulus values ordered among themselves. but in this first condition, we will put at random, because it is impossible to make an evaluation
                        if len(unstructured) == 2: return unstructured[0][0], rang_SM[0], unstructured[0][0]
                        if len(unstructured) >= 3: return unstructured[0][0], unstructured[int(len(rang_SM)/2)][0], unstructured[0][0]
                    if len(rang_SM) == 2: return rang_SM[0], unstructured[0][0], rang_SM[1]
                    if len(rang_SM) >= 3: return rang_SM[0], rang_SM[int(len(rang_SM)/2)], rang_SM[-1]
        # ***************************************************************************************************************

    def val_idx_convert(df,val,i):
        val_idx = None
        if val != None:
            val_idx = df[val][i]
            if 'gpa' in val.lower(): val_idx = float(rx.findall(str(val_idx))[0]) * 1000 # convertion, GPa = 1000 MPa
            if "G'" in val: val_idx = 2*(1+0.4)*val_idx # E' = 2(1+ν)G', ou v £ [0.3;0.5] # https://fr.wikipedia.org/wiki/Coefficient_de_Poisson
        return val_idx
        #***************************************************************************************************************

    for i in range(len(df)):
        col_list=[]
        for col in df.columns:
            if 'err' not in col:
                if is_float(df[col][i])==True :col_list.append(col)
        Tg = identified_Tg_col(col_list)
        tg_val = None
        if Tg != None:
            num = rx.findall(str(df[Tg][i]))
            if len(num)!=0: tg_val= float(num[0])
        SM_Av_Tg, SM, SM_Ap_Tg = identified_SM(col_list,tg_val)
        if SM_Av_Tg !=None or SM !=None or SM_Ap_Tg !=None: #and SM !=None
            print(SM_Av_Tg, SM, SM_Ap_Tg,'\n','------------------------------------------------')
            SM_Av_Tg_val = val_idx_convert(df, SM_Av_Tg, i)
            SM_Ap_Tg_val = val_idx_convert(df, SM_Ap_Tg, i)
            SM_val = val_idx_convert(df, SM, i)
            new_row = ({'Ref': df['Ref'][i], 'Samples': df['Samples'][i], 'Vectors': df['Vectors'][i], 'Tg (°C)': tg_val,'SM_Av_Tg (MPa)':SM_Av_Tg_val, 'SM (MPa)':SM_val, 'SM_Ap_Tg (MPa)':SM_Ap_Tg_val, 'Attributs': str(SM_Av_Tg) +'/'+ str(SM)+ '/'+ str(SM_Ap_Tg) })
            df_Tg_Modulus.loc[len(df_Tg_Modulus)] = new_row
            df_Tg_Modulus.to_excel("/Users/tchagoue/Documents/AMETHYST/Code_git/AMETHYST/Data_Mining/Storage_Tg/Tg_SM.xlsx")
    print(df_Tg_Modulus)
    print(len(df), len(df.columns))
    return df_Tg_Modulus

Dataframe = Tg_Modulus_extraction(file_csv)

#***********************************************************************************************************************
                                                        #End#
#***********************************************************************************************************************
# The consolidated view of the functional dependency function attributes is in the excel file Tg_SM.xlsx
#***********************************************************************************************************************






