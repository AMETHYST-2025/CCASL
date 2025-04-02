import numpy as np
import pandas as pd
import re
import nltk
import os
from chemdataextractor.doc import Paragraph, Heading
#pd.options.mode.copy_on_write = True
nltk.download('stopwords')
import ast

path = '/Users/tchagoue/Documents/AMETHYST/Datas/Normalized_CSV2/'
numeric_const_pattern = '[-+]? (?: (?: \d* \. \d+ ) | (?: \d+ \.? ) )(?: [Ee] [+-]? \d+ ) ?'
rx = re.compile(numeric_const_pattern, re.VERBOSE)
name='j.polymdegradstab.2013.02.008'
# **********************************************************************************************************************
skip_word = ['E-0-CNT', 'E-0.5-CNT', 'E-2-CNT', 'E-1-CNT', 'E-1.5-CNT', 'CNT-PD-10', 'CNT-PD-5', 'CNT-PD-20','NH2','SPE-TA1.0',
             'Gly-HPO', '44DDSb', '44DDS', '33DDS', 'PMI-HSi', 'PT-30', 'T7-Ph- POSS', 'Na-Ph- POSS', 'CF-PO (OPh)2',"3,3'DDS",
             'CF-POPh2', 'CF-PPh2', 'D-P-A', 'CF-PO(OPh)2', 'BA-CHDMVG', 'Tetra-DOPO', 'MDH', 'AlPi', 'BODIPY', 'ZnPi',
             'ATH','G1-PU','G2-PU','G3-PU','G4-PU','SMEP','GO-NH3','Na-Ph-POSS','T7-Ph-POSS','B1D9',' B2D8','SiO2',
             'jER828', 'jER1001', 'GO-c', 'Graphenit-ox', 'Graphenit-Cu', 'Cu2+-GO', 'Cu-rGO', 'CuO-GNS', 'BPA-BPP',
             'BODIPY-MXene', 'BODIPY', 'GO- PPD', 'PMI-HSi', 'IBOMA', 'CTA', 'ISOMA', 'APP@ATNi', 'NPES-901','NH2',
             'DOPO-PHE-P', 'DPO-PHE-P', 'PEI-APP', 'HPCP', 'm-Phenylenediamine', 'TRGH-700', 'TRGH-1000', 'TRGH-2000',
             'TRGB-2000', 'TRGB-1000','EB-40','SrF2','PA6','PA6/GO-0.5','PA6/GO-1.0','RTM6','A-PA6','PD-rGO','LPP-MoSe2',
             'TRGB-700', 'D230', 'T403','LDH-CD-Ferr']
# "skip_word" is use so that it is not segmented, when partitioning the components, and that the special characters of these compounds
# do not interfere with occurrence statistics.

Chemical_Componemt = ['CNT', 'E-0-CNT', 'E-0.5-CNT', 'E-2-CNT', 'E-1.5-CNT', 'E-1-CNT', 'CNT-PD-10', 'CNT-PD-5',
                      'CNT-PD-20', 'BPOPA', '44DDSb', '44DDS', '33DDS', 'APP', 'LHP', 'PMI-HSi', 'HS', 'LHP','LDH-CD-Ferr',
                      'HHPP', 'DHB', 'DOPO', 'DGEBA', 'HBP', 'PT-30', 'DIB', 'jER1001', 'GO- PPD', 'Tetra-DOPO', 'APP',
                      'ATH', 'MDH', 'AlPi', 'FGO','2E4MZ','EB-40','SiO2','SrF2','SMEP','GO-NH3','B1D9','B2D8','B3D7','NH2',
                      'CF-PO (OPh)2', 'CF-PO(OPh)2', 'TA', 'CF-PPh2', 'CF-POPh2', 'BA-CHDMVG', 'jER828', 'Cu2+-GO','D301',
                      'Cu-rGO', 'CuO-GNS', 'BPA-BPP', 'BODIPY', 'ZnPi', 'IBOMA', 'CTA', 'ISOMA', 'GDP', 'DCPD230','PD-rGO',
                      'DOPO-PHE-P', 'DPO-PHE', '3F', '2MI', 'EP', 'EPON 826', 'EPON826', 'EPON 828','PA6/GO-0.5','PA6/GO-1.0',
                      'EPON828', 'AESE', '[Dmim]Tos', 'TAS', 'o-DAMP', 'PMSE', 'PBI', 'DPCG', 'POBDBI', 'GO-c', 'CNT',
                      'MoSe2','G1-PU','G2-PU','G3-PU','G4-PU', #'EP1', 'EP2', impact sur j.eurpolymj.2011.10.017,  pCBT/EP 1%
                      'Graphenit-ox', 'Graphenit-Cu', 'GO', 'BODIPY-MXene', 'D230', 'BE188', 'NPES-901', 'PA650','PA6',
                      'P', 'PEI-APP', 'HPCP', 'ATPB', 'PH', 'm-Phenylenediamine', 'Ancamine2049', 'CoSA', 'ATNi','RTM6',
                      'DTA', 'IPD', 'PACM', 'DDS', 'DDM', '3DCM', 'MDEA', 'MCDEA', 'TMAP', 'D400', 'HMDA', 'DGEBU','LPP-MoSe2',
                      'TGDDM', 'TGAP', 'DETDA', 'PMMA', 'DER332', 'DER331', 'PMI-HSi', '2OA', '20A', 'MU22','A-PA6',
                      '3D-C-BNNS','3D-BNNS','3D-BN','H2TPMP','SPE-TA1.0',
                      'TRGH-700', 'TRGH-1000', 'TRGH-2000', 'TRGB-2000', 'TRGB-1000', 'TRGB-700', 'T403', 'T-403']

# "Chemical_Componemt" is the list of chemical elements to isolate so that their numbers are not considered as
# weights. It also makes it possible to determine the weights stated in a factorized way, e.g.: DGEBA-25%.
# Note that: for two similar elements, start with the longest: Example 33DDS is stated before the DDS, otherwise,
# 33 will be considered as a weight

# Exemple de composé identique : BE188 = diglycidyl ether of bisphenol-AðEEW 1⁄4 188Þ = NPES-901 =  diglycidyl ether of bisphenol-AðEEW 1⁄4 500Þ.
# EPON 826 = DGEBA =  'EPON 826' = 'EPON826' = 'EPON 828' = 'EPON828'
# Jeffamine = T403 = T-403
# **********************************************************************************************************************

rank_Chemical_componemt = {}
for ele in Chemical_Componemt:
    rank_Chemical_componemt[ele] = len(ele)
# you must arrange the chemicals_componemt by decreasing length, to avoid DDS being recognized instead of 33DDS for example
rank_Chemical_componemt = dict(sorted(rank_Chemical_componemt.items(), key=lambda item: item[1], reverse=True))
Chemical_Componemt = rank_Chemical_componemt

def ChemDataExtractor(text):
    text = re.sub(u'[^\u0020-\uD7FF\u0009\u000A\u000D\uE000-\uFFFD\U00010000-\U0010FFFF]+', '', text)
    p = Paragraph('u"' + text + '"')
    T = p.cems
    list = []
    for ele in T:
        print(ele)
        list.append(ele)
    return list
    # ******************************************************************************************************************

def replace_col(k,df, new_name):
    df.rename(columns={df.columns[k]: new_name}, inplace=True)
    return df
    # ******************************************************************************************************************

def delete(value,ele_list):
    for ele in ele_list:
        if value.find(ele) != -1:
            value = value.replace(ele, "")
    return value
    # ******************************************************************************************************************

def Replace(value,strings,R_values):
    for i in range(len(strings)):
        value=value.replace(strings[i],R_values[i])
    return value
    # ******************************************************************************************************************

def list_find_key(value, key):
    list = []
    for i in range(len(value)):
        if value.find(key, i, i + 1) != -1: list.append(i)
    return list
    # ******************************************************************************************************************

def Replace_by(value, strings, val):
    for i in range(len(strings)):
        value = value.replace(strings[i], val)
    return value
    # ******************************************************************************************************************

def rename_epoxy(value):
    skip = 0
    for ele in Chemical_Componemt:
        if ele in value and ele not in ['EP', 'P']:
            skip = 1
    if skip == 0:
        value = Replace_by(value,
                           ['Neat epoxy resin','neat epoxy','Epoxy resin','EP-NEAT', 'Neat EP', 'Neat epoxy resins',
                            'epoxy resin', 'Neat Epoxy', 'Neat epoxy',
                            'Neat', 'epoxy',
                            'Epoxy', 'neat'], 'EP')
    else:
        value = Replace_by(value, ['filled', 'neat'],
                           '')  # for cases such as : EP/D230-29 neat ['EP', 'D230(29)EP']  j.eurpolymj.2016.09.022
    value = delete(value, ['Neat', 'Pure', ' '])
    return value
    # ******************************************************************************************************************

def eliminate_ref(value):
    #define a code to identify and eliminate bibliographic references,
    #Ex : in j.polymdegradstab.2021.109629_5_0, the sample has a number as reference and it can be taken as weight DOPO-EP1 [40]
    T = rx.findall(value)
    if len(T)!=0:
        if value.find('['+T[-1]+']')==len(value)-2-len(T[-1]): value=value.replace('['+T[-1]+']','') # verifier que l'element en crochet'[]' se trouve a la fin de value
    value = value.replace('//','/')
    return value
    # ******************************************************************************************************************

def double_percentage(value,ponct): # see example here j.polymdegradstab.2020.109134_2_0
    list_pourcentage = list_find_key(value,'%')
    list_ponct = list_find_key(value, ponct)
    try:
        if len(list_pourcentage)>=2 and len(list_pourcentage)>len(list_ponct) and list_ponct[0]<list_pourcentage[0]:
            # to select cases where the author omits the division symbol between two compounds, Example:UP-20% APP 5% ATH must be : UP-20% APP- 5% ATH
            # new observation: this condition, to be strict must be : len(list_pourcentage)>len(list_ponct) + 1 (12 Mars 2024)
            T = rx.findall(value)
            for ele in T:
                #ele = ele.replace('-','')
                if value.find(ele+'%') != -1:
                    for ele in [ele.replace('-',''),ele]: # to avoid confusion with elements that contain '-' as basic punctuation
                        if value.find(ponct+ele+'%')==-1 or value.find(ponct+' '+ele+'%')==-1:
                            value = value.replace(ele+'%',ponct+ele+'%')
                            value = value.replace('--','-') # to remove the duplicate create in the case where the division symbol is '-'
                            break
    except:
        value=value
    return value
    # ******************************************************************************************************************

def samples_normalization(df, Chemical_Componemt=Chemical_Componemt, skip_word = skip_word):
    list_vectors = []
    try:
        df['Vectors'] = df.loc[:, ['Samples']]
    except:
        replace_col(0, df, "Samples")
        df['Vectors'] = df.loc[:, ['Samples']]
        print('Sample detection to review for this!!!')
    def find_int(value):
        def search_chemical(value):
            for chemical in Chemical_Componemt:
                if value.find(chemical)!=-1:
                    return 1, chemical
            return 0, 'No_chemical'
        for esp in ['   ','  ',' ']: # The goal is to eliminate the spaces at the beginning and end of the string, risk becoming obsolete if I delete all the spaces (15/03/24)
            if value.find(esp)==0: value = value[len(esp):]
            if value[-len(esp):] == esp :value = value[:-len(esp)]
        value = value.replace(' ','')
        pto = list_find_key(value, '(') ; ptf = list_find_key(value,')')
        if abs(len(ptf)-len(pto))==1: # This algorithm allows us to eliminate unstructured parentheses, examples:  EP + (MPZnP + AIO(OH)) becomes : [EP, (MPZnP, AIO(OH))]
            if value[-1] == ')' and len(ptf)>len(pto) : value = value[:-1]
            if value[0] == '(' and len(pto)>len(ptf)  : value = value[1:]

        #value = Replace_by(value,['wt .%','wt %','wt%','Wt%'],'wt.%')
        T=rx.findall(value); stat=0;
        approche_2, Molecule = search_chemical(value)

        if len(T)==2 : # To convert the ',' in '.', Examples : 0,25 GO/EP/TETA   =  [0,25GO, EP, TETA]   ref : j.compscitech.2016.10.014
            value=value.replace(T[0]+','+T[1],T[0]+'.'+T[1])
            T = rx.findall(value)

        if len(T)!=0 and approche_2 ==0:
            h=len(T[0])#i
            To = T[0].replace('-', '')
            ratio = str(round((float(To)/100),4))
            pos=value.find(T[0])  #i
            if (value[pos+h:]).find('%')!=-1:
                value = value[:pos]+'('+ ratio +')'+value[pos+h:];
                value=delete(value,['wt.%','wt','%']); stat=1
                value = value.replace('(('+ratio+'))','('+ratio+')')# To avoid this : [CNFs((0.01))/EP]  j.matchemphys.2009.07.045
            elif (value[pos+h:]).find('£')!=-1:
                value = value[:pos]+'('+To+')'+value[pos+h:];
                value=delete(value,['£']); stat=1
            else :
                print(value)
                value = value[:pos] + '(' + To + ')' + value[pos + h:] # i
                print(value)
                print('ici')

        if len(T)!=0 and approche_2 == 1: # case where the chemical element has numbers in its name
                #------ list of derived submolecules----------
                #Ex : EP, EP-1, EP-2...
                sous_molecules = []
                for i in range(10):
                    sous_molecules.append(Molecule + str(i))
                    sous_molecules.append(Molecule+'-'+str(i))
                # ---------------------------------------------
                value_f= value.replace(Molecule,'')
                T = rx.findall(value)
                T2 = rx.findall(value_f)
                value_fi=value
                if len(T2) !=0:
                    ratio = str(round(float(T2[0].replace('-', '')) / 100, 4))
                    if len(T2) >1 : # PH2(0.3), for such a case, where PH is the molecule, T2 has two values : 2 et 0,3.
                        if value.replace('(' + T2[1] + ')','') in sous_molecules :
                            value_fi = value
                    elif value.find('('+T2[0]+')') ==-1 : # This condition avoids doubling the parentheses, if the original version is already standardized
                        if value.find('£') !=-1:
                            pos=value.find(T[0]) ; To = T[0].replace('-','') # To allows you to eliminate the '-' sign which is introduced into the weight of the molecule
                            value_fi = value[:pos] + '(' + To + ')' + value[pos + len(T[0]):];
                            value_fi = delete(value_fi, ['£'])
                        elif value not in sous_molecules: # this makes it possible to avoid considering the DGEBA-1 as DGEBA(1) ...
                            value_fi = value.replace(T2[0],'('+T2[0].replace('-','')+')')
                    if value_f.find('wt.%') != -1: value_fi = value.replace(T2[0] + 'wt.%','(' + ratio + ')')
                    elif value_f.find('wt%') != -1: value_fi = value.replace(T2[0] + 'wt%', '(' + ratio + ')')
                    elif value_f.find('wt') != -1: value_fi = value.replace(T2[0] + 'wt','(' + ratio + ')')
                    elif value_f.find('%')!=-1: value_fi = value.replace(T2[0]+'%','('+ratio+')')
                    value_fi = value_fi.replace('((' + ratio + '))', '(' + ratio + ')')  # To avoid double parentheses : [CNFs((0.01))/EP]  j.matchemphys.2009.07.045
                else: value_fi = Molecule
                value= value_fi

        return  value
        #***************************************************************************************************************

    def count_ponct(value,ponc):
        count=0; exclude =[]
        for word in skip_word:
            findd = value.find(word)
            if findd != -1:
                for i in range(len(word)):
                    exclude.append(findd+i)
        for i in range(len(value)):
            if value.find(ponc, i, i + 1) !=-1 and i not in exclude: count+=1
        return count
        # ***************************************************************************************************************

    def main_ponctuation(df):
        # Here we determine the main punctuation which serves as a pivot symbol to the structure of the compound, Example : < "/" dans : EP/DDM/DOPO >
        minus = 0; under_score = 0; slash = 0; dpts = 0; plus=0; arobase = 0; ponct = '/'; ponct2=''; ponc_list={}
        for k in range(len(df)):
            value = str(df['Vectors'][k])
            value = value.replace("@", "/") # same in the function 'split'
            #if value.find('Cu2+-GO') ==-1 : value = value.replace("+", "/")
            Minus = count_ponct(value, '-'); Slash = count_ponct(value, '/'); Under = count_ponct(value, '_'); Plus = count_ponct(value,'+'); Dpts = count_ponct(value,':'); Arobase = count_ponct(value,'@')
            minus += Minus; slash += Slash; under_score += Under; plus+=Plus; dpts+=Dpts; arobase+=Arobase
            plus+=Arobase # consider '@' like the pivot symbol, comparable to the '/'
        if plus > slash and plus > under_score and plus>minus:
            ponct = '+'; ponc_list['+']=plus
        if slash >= minus-1 and slash >= under_score and slash !=0:
            ponct = '/' ; ponc_list['/'] = slash
        if minus > slash+1 and minus >= under_score and minus !=0:
            ponct = '-' ; ponc_list['-'] = minus
        if under_score > slash and under_score > minus and under_score != 0:
            ponct = '_' ; ponc_list['_']= under_score
        if plus == minus and slash+under_score ==0: # We can decide to favor the '+' on the  '-' in case of equality : DGEBA + 20 wt % LPN-EP
            ponct = '+'
        # The two additional additional conditions appear on March 11, 2024, following the observation that it is necessary to take into account the ':', dpts: DGEPEG:DGEBA/APEG:IPDA/5.0 wt % SiO2 , j.ijadhadh.2019.102430
        if arobase > slash and arobase > plus and arobase > minus and arobase > under_score and arobase!=0:
            ponct = '@' ; ponc_list['@']= arobase
        #if dpts > slash and dpts>plus and dpts>minus and dpts>under_score and dpts!=0 : # dpts should only be considered as a secondary symbol, because other considerations create a degeneracy on sp_norm2
            #ponct = ':' ; ponc_list['@']= dpts # cause pbs ici : 2OA:IBOMA + CTA* (85:15),
        if plus/(slash+1)>=0.7 and slash>2: ponct2 = '+' # # Si 70% of characteristics symbols are '+', consider creating secondary pivot punctuation.  To go further, consider in the future a list of punctuations in order of priority, taking into account good segmentation (detection of basic chemical elements present in the dictionary, before or after the symbol)
        if dpts/(slash+1) >= 0.5 and slash>2 : ponct2 = ':'
        if arobase/(slash+1) >= 0.5 and slash > 2: ponct2 = '@'
        ponc_list = dict(sorted(ponc_list.items(), key=lambda item: item[1], reverse=True))
        return ponct, ponct2, ponc_list
        # ***************************************************************************************************************

    def cas_particulier(value):
        # For particular labels such as EP-TA-1.25, here we must have EP/TA(1.25), however this approach does not work in this example : EP@PCTP-4
        value =rename_epoxy(value)
        number = rx.findall(value); e=0; no=0; l_excep = 0
        for exclude in skip_word:
            if value.find(exclude)!=-1:
                l_excep = value.find(exclude) + len(exclude)-1 # Prevent chemical compounds like PT-30 from being converted to PT(30)
        if len(number) != 0:
            if len(number) == 1 and value[:-len(number[0].replace('-',''))-1] + '-' + number[0].replace('-','') == value : no = 1  # Do not segment if there is only one molecule. ruler : si on a SMEP + (-1) = SMEP-1, j.compscitech.2021.108899_2_0
            for num in number:
                if value[:-len(num) - 3] + '-' +num.replace('-','') + 'wt%' == value: no = 1 # Avoid segmentation here EP(PA6)-0.5wt%,j.compscitech.2018.12.023_6_0
                num = num.replace('-','') # Remove the hyphen to make sure that all the 'num's don't have one
                search = value.find('-'+num) #
                #if value.find('EP-'+num)!=-1:no=1
                if search!=-1 and value.find('EP-'+num)==-1 and len(value)-1 == search +len(num) : e+=1 # 1st cond: existence of possible proportions in -30, 2nd cond: do not consider the epoxy types (EP), 3rd cond: only take into account the last elements, 4th cond: do not consider anything if it is the nomenclature of a compound
                if len(value)-1>search+len(num):
                    if search!=-1 and value.find('EP-'+num)==-1 and value[search + len(num) + 1] in ['', ' ']:e+=1 # Last condition to clarify that we are dealing with the last element EX: EP-TA-1.25
                if e !=0 and l_excep ==0 and len(num)>1: value = value[:search] + value[search:].replace('-'+num,num+'£') # DGEBA-1.5 = DGEBA 1.5% => DGEBA(1.5) #.replace('-'+num,num+'%')
                if value == num+'-'+value[len(num)+1:] : # To eliminate cases :  0.1-GNP-Epoxy becomes :  0.1GNP-EP
                    value = value[:value.find('-')]+value[value.find('-')+1:]
                    print(value)
        return value, no
       #****************************************************************************************************************

    def split(value,ponctualion):
        dict_ponct = []; splitted=[]; skip_index=[]

        # ---------------------Cas_particulier : 3%-Graphenit-ox-Epoxy -> 3%[Graphenit-ox]-Epoxy------------------------
        indx = value.find('%-')
        if indx != -1 and indx <=4 :
            value = value[:4].replace('-','')+value[4:]
        if len(value) != 0 : # fix the bug that occurs in this scenario : EP/DDS/DIB-0.25., which disrupts the weight conversion (0.25)
            if value[-1] == '.' : value = value[:-1]
        # --------------------------------------------------------------------------------------------------------------

        dict_ponct.append(0)  # The position of the first letter
        value,no = cas_particulier(value)
        if ponct2 != '': value = value.replace(ponct2, ponctualion)  # Transform the '+' symbol into '/', after observation, this could make things easier # 12 Oct 2023, first implementation on March 11, 2024
        for ele in skip_word:
            if value.find('(' + ele + ')') != -1:
                value = Replace(value, ['(' + ele + ')'], [ele])
            if ele in value:
                for i in range(len(ele)):skip_index.append(value.find(ele)+i)

        # To avoid segmentation of the last dash: Epoxy-DDM-1
        tiret_fin = 1000
        numbers = rx.findall(value)
        if len(numbers) !=0:
            for numb in numbers :
                numb = numb.replace('-', '')
                for charactar in ['-','/','_']:
                    if value in [value[:-len(numb)-1]+charactar+numb, value[:-len(numb)-2]+charactar+numb+'%', value[:-len(numb)-4]+charactar+numb+'wt%']:
                        tiret_fin = list_find_key(value,charactar)[-1]
        for i in range(len(value)):
            if value.find(ponctualion, i, i + 1) != -1 and i not in skip_index and i != tiret_fin: dict_ponct.append(i+1)

        dict_ponct.append(len(value))  # The position of the last letter

        if len(dict_ponct) >= 3:
            for j in range(len(dict_ponct) - 1):
                value1_n = value[dict_ponct[j]:dict_ponct[j + 1]]
                a=0;skip_index_=[]
                for ele in skip_word:
                    if ele in value1_n:
                        for i in range(len(ele)): skip_index_.append(value1_n.find(ele) + i)
                for i in range(len(value1_n)):
                    if value1_n.find(ponctualion, i, i + 1) ==i and i not in skip_index_: value1_n = value1_n[:i]+value1_n[i+1:]
                value1_n = rename_epoxy(value1_n)
                value1_n = find_int(value1_n) #Identify the weights associated with the compound and put them in the form of the nomenclature : (poids).
                splitted.append(value1_n)

        # The case where the creation of the vector will destroy a molecule of the type : PDA(DOP-O)2 != ['PDA(DOP' ; 'o)2']
        if len(dict_ponct)==3 and (len(splitted[-1])==1 or len(splitted[-1])==0) : splitted=[] # Case where we have only one punctuation, example : EP-24, In this case, we have ['EP-24'] et not ['EP','24'], 11 Mars 2024 : take into consideration the case where 'EP/' becomes : [EP, ]

        # -------------------------------------------------------------------------------
        idx_p=1 # This algorithm makes it possible to avoid confusion in cases such as those
        # generated by samples of the type : (BA-CHDMVG)2o-(jER828)20-(jER1001)60
        while value.find(')')!=-1 and idx_p < len(dict_ponct)-1:
            if value.find('(')<= dict_ponct[idx_p] and value.find(')') <= dict_ponct[idx_p]:
                value = value[value.find(')'):]
            idx_p+=1 # We iterate over the punctuation indices, except for the 0 and the last
            # (len()-1), because the last ones, are not punctuation clues
        #-------------------------------------------------------------------------------

        if (value[:value.find(ponctualion)]).find('(') !=-1 and \
                (value[value.find(ponctualion):]).find(')') !=-1 : splitted=[]
        if len(dict_ponct) == 3 and  no == 1 and tiret_fin ==1000 and '£' not in value : splitted=[]
        return splitted
        #***************************************************************************************************************

    ponct, ponct2, ponct_list = main_ponctuation(df)
    def num_detection(value):
        T = rx.findall(value)
        if value.find('(') != -1 and value.find(')') != -1 and len(T) != 0:
            for T1 in T:
                if value.find('('+T1+')')!=-1:
                    T=T1
                    value2 = '(' + T + ')'
                    value = delete(value, [value2])
                    #T = format(float(T), '.4f')
                    T = float(T)
                    break
        else: T=1
        return value, T
        #***************************************************************************************************************

    def to_Sp_norm(df): # For special cases like  UP/epoxy-PH: 70/30-BADP , UP/epoxy-PH: 50/50, Epoxy-PH: RDP, PH: RDP
        P_ = 0; Non =0; decision_n = 0
        for k in range(len(df)):
            count_ok = 0
            value = str(df['Samples'][k])
            list_slash = list_find_key(value,'/')
            p1 = value.find(':')
            if p1 != -1 and len(list_slash) != 0: P_+=1 # Presence of the two basic symbols
            if p1 !=-1 :
                for idx in list_slash:
                    if idx-3 <0 :
                        start_idx = 0
                    else : start_idx = idx-3
                    valeur_gauche = rx.findall(value[start_idx:idx]) # UP/PH2:70/30, 70 in this case
                    valeur_droite = rx.findall(value[idx + 1:idx + 3]) # UP/PH2:70/30, 30 in this case
                    if len(valeur_droite)!=0 and len(valeur_gauche)!=0:
                        if value.find(rx.findall(value[start_idx:idx])[0]+'/'+ rx.findall(value[idx+1:idx+3])[0]) !=0 : count_ok+=1
            if count_ok ==0 and len(list_slash) !=0 : Non +=1
            len_2pts = len(list_find_key(value,':'))
            if len_2pts >=2 :
                return 0 # in order to eliminate all other cases, notably : DGEPEG:DGEBA/APEG:IPDA/ 5.0 wt% SiO2, j.ijadhadh.2019.102430
        if P_ != 0 and Non == 0: decision_n =1 ; print('jdlfnjsq')
        return  decision_n
        # ***************************************************************************************************************

    def to_Sp_norm2(value,ponct): # For special cases like j.polymdegradstab.2013.12.021, avec :" 2OA:IBOMA: ISOMA + CTA* (85:10:5)" ponct, is the main punctuation (special character)
        value_final = value
        def count_sym(value,symbole):
            nbr=0; position= []
            for i in range(len(value)):
                if value[i]==symbole: nbr+=1; position.append(i)
            return nbr, position
        list_2_pts = []; list_pth_ouv = []; list_pth_fer = []
        #important pretreatment, for the considerations made below:
        if len(value)>4 :
            if value[0]==' ': value=value[1:]
        value = Replace_by(value,[' :',': ','  :',':  '],':')
        for i in range(len(value)):
            if value[i]==':':list_2_pts.append(i)
            if value[i] == '(': list_pth_ouv.append(i)
            if value[i] == ')': list_pth_fer.append(i)
        if len(list_pth_fer) == 1 and len(list_2_pts) !=0 :
            try:
                value_dans_pth = value[list_pth_ouv[0]:list_pth_fer[0]]
                value_hors_pth = value[:list_pth_ouv[0]]+value[list_pth_fer[0]:]
                nbr_in, pos_in = count_sym(value_dans_pth,':')
                nbr_out, pos_out = count_sym(value_hors_pth, ':')
                poids = rx.findall(value_dans_pth)
                unité = '';total=0
                for i in poids:
                    total+=int(float(i))
                if total==100: unité = '%'
                if len(list_pth_ouv)==len(list_pth_fer) and nbr_out==nbr_in and len(poids)==nbr_in+1:
                    value_final=''; list_seg=[0]
                    list_seg += list_2_pts[:nbr_out]
                    list_seg.append(value.find(' ')) # in order to cut out the last element that cannot be identified by the presence of ':'
                    for i in range(len(poids)):
                        if i != len(poids)-1 : value_final += value[list_seg[i]:list_seg[i+1]]+' '+poids[i]+unité+ponct
                        if i == len(poids)-1 : value_final += value[list_seg[i]:list_seg[i+1]]+' '+poids[i]+unité # the presence of punctuation increases an additional compound
                        value_final = value_final.replace(':','')
                    value_final+= value[list_seg[-1]:list_pth_ouv[0]] # to increase the remaining portion of the text :2OA:IBOMA: ISOMA + CTA* (85:10:5) must become : 2OA85%IBOMA10%ISOMA5%+ CTA*
            except:
                value_final = value
        return  value_final
        # ***************************************************************************************************************

    def to_Sp_norm3(value,ponct):
        #ER-DiPrOHIm-3% , to avoid this : [ER, DiPrOHIm, (0.03)], j.eurpolymj.2021.110296_6_0
        list_ponct = list_find_key(value,ponct)
        T= rx.findall(value)
        if len(list_ponct) !=0 and len(T)!=0 and ponct=='-':
            for num in T:
                num = num.replace('-','')
                for ele in ['wt.%','wt','%']:
                    if value[-(len(num)+len(ele)+1):] == '-'+num+ele :
                        value = value[:list_ponct[-1]] +value[-(len(num)+len(ele)):] # we transform ER-DiPrOHIm-3% in ER-DiPrOHIm3%, the rest of the algorithm will do the trick
                        break
        print(value)
        return value
        # **************************************************************************************************************


    def to_Sp_norm4(value,ponct): # For special cases like j.eurpolymj.2004.03.005, avec : 40/40/20 (BTTEMAT/TEGDMA/TMPTMA)
        list_slash = list_find_key(value,'/')
        list_pth_ouv = list_find_key(value,'(')
        list_pth_fer = list_find_key(value,')')
        value_final = value
        if len(list_pth_fer) == 1 and len(list_slash) !=0 and len(list_pth_ouv) == len(list_pth_fer):
            value_dans_pth = value[list_pth_ouv[0]:list_pth_fer[0]]
            value_hors_pth = value[:list_pth_ouv[0]]+value[list_pth_fer[0]:]
            poids = rx.findall(value_hors_pth)
            unité = '';total=0
            for i in poids:
                total+=int(float(i))
            if total==100: unité = '%'
            if len(poids)==len(list_slash)/2+1:
                print(poids)
                value_final = ''
                for i in range(len(poids)):
                    if i==0 : value_final += value[list_pth_ouv[0]+1:list_slash[int(len(list_slash)/2)+i]]+' '+poids[i]+unité+ponct
                    if i!=0 and i!=len(poids)-1 : value_final += value[list_slash[int(len(list_slash)/2)+i-1]+1:list_slash[int(len(list_slash)/2)+i]]+' '+poids[i] +unité+ponct # la présence de la ponctuation augmente un composé supplementaire
                    if i==len(poids)-1 and i !=0: value_final += value[list_slash[int(len(list_slash)/2)+i-1]+1:list_pth_fer[-1]]+' '+poids[i]+unité
        print(value_final)
        return  value_final
        # **************************************************************************************************************

    def to_Sp_norm5(value,ponct): # For special cases like "E/M-Mt(0.05:1)"  [E, M-Mt((0.05), (1)]  j.compositesb.2018.12.028_8_0
        list_2pts = list_find_key(value,':')
        list_pth_ouv = list_find_key(value,'(')
        list_pth_fer = list_find_key(value,')')
        list_slash = list_find_key(value,'/')
        value_final = value
        if len(list_pth_fer) == 1 and len(list_2pts) !=0 and len(list_pth_ouv) == len(list_pth_fer) and list_pth_ouv[0]<list_2pts[0] and list_pth_fer[0]>list_2pts[-1]:
            value_dans_pth = value[list_pth_ouv[0]:list_pth_fer[0]]
            value_hors_pth = value[:list_pth_ouv[0]]+value[list_pth_fer[0]:]
            poids = rx.findall(value_dans_pth)
            unité = '';total=0
            for i in poids:
                total+=int(float(i))
            if total==100: unité = '%'
            if len(poids)==len(list_2pts)+1 and len(value_hors_pth)>2: # last condition to avoid the case where we only have (1:1)
                print(poids) #E/M-Mt(0.05:1)
                value_final = ''
                for i in range(len(poids)):
                    if i==0 : value_final += value[0:list_slash[i]]+' '+poids[i]+unité+ponct
                    if i!=0 and i!=len(poids)-1 : value_final += value[list_slash[i-1]+1:list_slash[i]+1]+' '+poids[i] +unité+ponct # The presence of punctuation increases an additional compound
                    if i==len(poids)-1 and i !=0: value_final += value[list_slash[i-1]+1:list_pth_ouv[0]]+' '+poids[i] +unité
        print(value_final)
        return  value_final
        # **************************************************************************************************************

    decision_sp = to_Sp_norm(df)
    def Special_normalization(df,k, decision_n): # For special cases like  UP/epoxy-PH: 70/30-BADP , UP/epoxy-PH: 50/50, Epoxy-PH: RDP, PH: RDP
        if decision_n == 1:
            value = df['Samples'][k]
            value= rename_epoxy(value)
            ponc = value.find(':')
            ponc1 = value[:ponc].find('/')
            ponc2 = value[ponc:].find('/')
            if ponc1 != -1 and ponc2 != -1:
                val_1 = rx.findall(value[ponc + ponc2 - 3:ponc + ponc2])[0]
                val_2 = rx.findall(value[ponc+ponc2:ponc+ponc2+3])[0]
                poids_1 = '('+ str(float(val_1)/100) + ')'  #str(float(T[i])*0.01)
                poids_2 = '('+ str(float(val_2)/100) + ')'
                value_mod = value[:ponc1]+ poids_1 + '/'+ value[ponc1+1:ponc] + poids_2
                ponc3 = value[ponc+ponc2:].find('-')
                if ponc3 !=-1:
                    value_mod = value_mod + '/' + value[ponc+ponc2+ponc3+1:]
                df['Vectors'][k] = split(value_mod,'/')
            else :
                df['Vectors'][k] = split(value,':')
                if len(df['Vectors'][k]) == 0 : df['Vectors'][k] = [value]
        #***************************************************************************************************************

    for k in range(len(df)):
        value=str(df['Samples'][k])
        value = eliminate_ref(value)
        value = double_percentage(value,ponct) # UP-20% APP 5% ATH
        value = to_Sp_norm2(value,ponct) # For cases : "2OA:IBOMA (85:15)"
        value = to_Sp_norm3(value,ponct) # For cases : "ER-DiPrOHIm-3%"  j.eurpolymj.2021.110296_6_0
        value = to_Sp_norm4(value,ponct) # For cases:  "40/40/20 (BTTEMAT/TEGDMA/TMPTMA)"
        value = to_Sp_norm5(value, ponct)  # For cases : "E/M-Mt(0.05:1)"  j.compositesb.2018.12.028_8_0
        value = delete(value,["Samples"])
        split_value=split(value,ponct)
        df['Vectors'][k] = split_value
        #df.loc[k, 'Vectors'] = split_value
        if len(df['Vectors'][k])==0: # If the element does not contain composition punctuation.
            Vec,no = cas_particulier(str(df['Samples'][k]))
            if len(Vec)>1:
                if Vec[-1]==ponct:Vec=Vec[:-1]
            Vec=find_int(Vec)
            Vec=delete(Vec,["'","[","]"," "])
            df['Vectors'][k]=[Vec]
        Special_normalization(df,k,decision_sp)  # manage cases with ":" and "/" like : "UP/epoxy-PH: 70/30-BADP"
        ratio_total = 0; vector_correction = []
        for ele in df['Vectors'][k]:
            ele, T = num_detection(ele)
            try :
                ratio_total+=int(T)
                vector_correction.append( ele + '(' + str(round((float(T)/100),4)) + ')' )
            except : None
            if ele not in list_vectors: list_vectors.append(ele)

        # To calculate ratios given in percentage without symbol '%', in special case like HHPP/DHB100 = [HHPP, DHB(100)],
        # HHPP should be 0. However, this is not mentioned. Therefore, the total will be 101, we settle this below:
        if ratio_total == 100 :
            df['Vectors'][k] = vector_correction
        if ratio_total == 101 and len(df['Vectors'][k]) ==2:
            for i in range(len(vector_correction)):
                vector_correction[i]=vector_correction[i].replace('(0.01)','(0)')
            df['Vectors'][k] = vector_correction

    print(list_vectors)
    list_vectors_ = list_vectors
    def Encode(df):
        for j in range(len(df)):
            encode = []
            valueD = {}
            for value in df['Vectors'][j]:
                value, T = num_detection(value)
                valueD[value]=T
            for i in range(len(list_vectors_)):
                if list_vectors_[i] in valueD:
                    encode.append(valueD[list_vectors_[i]])
                else:
                    encode.append(0)
            print(encode)
    Encode(df)
    print(ponct)
    return list_vectors,df
    # ******************************************************************************************************************

def inference(name):
    path_csv = path+name
    df = pd.read_csv(path_csv, index_col=0)
    print(df)
    list, df = samples_normalization(df)
    print(df[['Samples','Vectors','Ref']][:])
    df.to_csv('/Users/tchagoue/Documents/AMETHYST/Code_git/AMETHYST/Normalization/Test/'+name+'.csv')
    # ******************************************************************************************************************
#inference(name) # run to test.

# Epoxy resin data-industrial-sheet : https://www.westlakeepoxy.com/chemistry/epoxy-resins-curing-agents-modifiers/epoxy-tds/
# See this site for the list of anime and epoxy : https://pubchem.ncbi.nlm.nih.gov/#query=amine&annothits=Chemical%20and%20Physical%20Properties
