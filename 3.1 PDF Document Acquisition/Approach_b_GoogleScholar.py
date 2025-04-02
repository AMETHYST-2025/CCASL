import requests
import re, json
from time import sleep
from bs4 import BeautifulSoup
import pandas as pd
import random
from scidownl import scihub_download
from io import StringIO
import sys


headers = {
    'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36'}
"""headers = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:109.0) Gecko/20100101 Firefox/109.0',
    'Accept-Language':	'en-US,en;q=0.5',
    'Accept-Encoding':	'gzip, deflate, br',
    'Accept':'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
    'Referer' :	'http://www.google.com/' }"""

# This function retrieves information from the web page
def get_paperinfo(paper_url):
    response = requests.get(url, headers=headers)
    # check successful response
    if response.status_code != 200:
        print('Status code:', response.status_code)
        raise Exception('Failed to fetch web page ')

    # parse using beautiful soup.
    paper_doc = BeautifulSoup(response.text, 'html.parser')

    return paper_doc


def get_tags(doc):
    paper_tag = doc.select('[data-lid]')
    #cite_tag = doc.select('[title=Cite] + a')
    link_tag = doc.find_all('h3', {"class": "gs_rt"})
    #author_tag = doc.find_all("div", {"class": "gs_a"})

    return paper_tag, link_tag


def get_papertitle(paper_tag):
    paper_names = []

    for tag in paper_tag:
        paper_names.append(tag.select('h3')[0].get_text())
    return paper_names


def get_citecount(cite_tag):
    cite_count = []
    for i in cite_tag:
        cite = i.text
        if i is None or cite is None:  # if paper has no citatation then consider 0
            cite_count.append(0)
        else:
            tmp = re.search(r'\d+',
                            cite)  # its handle the None type object error and re use to remove the string " cited by " and return only integer value
            if tmp is None:
                cite_count.append(0)
            else:
                cite_count.append(int(tmp.group()))

    return cite_count


# function for the getting link information
def get_link(link_tag):
    links = []

    for i in range(len(link_tag)):
        links.append(link_tag[i].a['href'])

    return links


def get_doi(paper_tag):
    paper_tag = []

    for tag in paper_tag:
        paper_doi.append(tag.select('doi')[0].get_text())


def get_author_year_publi_info(authors_tag):
    years = []
    publication = []
    authors = []
    for i in range(len(authors_tag)):
        authortag_text = (authors_tag[i].text).split()
        year = int(re.search(r'\d+', authors_tag[i].text).group())
        years.append(year)
        publication.append(authortag_text[-1])
        author = authortag_text[0] + ' ' + re.sub(',', '', authortag_text[1])
        authors.append(author)

    return years, publication, authors


paper_repos_dict = {
    'Paper Title': [],
    'Year': [],
    'Author': [],
    'Citation': [],
    'Publication': [],
    'Url of paper': [],
}


# adding information in repository
def add_in_paper_repo(papername, year, author, publi, link):
    paper_repos_dict['Paper Title'].extend(papername)
    paper_repos_dict['Year'].extend(year)
    paper_repos_dict['Author'].extend(author)
    #paper_repos_dict['Citation'].extend(cite)
    paper_repos_dict['Citation'].extend(publi)
    paper_repos_dict['Url of paper'].extend(link)

    return pd.DataFrame(paper_repos_dict)

def take_doi(value):
    vec = []
    vec.append(0)
    for i in range(len(value)):
        if value[i]=='/':
            vec.append(i)
    doi = value[vec[-1] + 1:]
    if value.find('doi/')!=-1:
        doi = value[value.find('doi/')+5]
    if doi in ['',' ']:
        doi = value[vec[-2] + 1:]
        if doi in ['abs/']:
            doi = value[vec[-3] + 1:]
    if doi[-1]=='/': doi = doi[:-1]
    if doi[0] == '/': doi = doi[1:]
    return doi
PATH = "/Users/tchagoue/Documents/AMETHYST/CCASL/3.1 PDF Document Acquisition/"
"""
# Execute only once to create the file 'list_data_download.xlsx'
import pandas as pd
DF_DOI = pd.DataFrame(columns=['Journal_Title','link','DOI', 'state'])
DF_DOI.to_excel(PATH+'list_data_download.xlsx', index=True)
"""

def Download_SCI(paper,paper_type,out):
    tmp = sys.stdout
    my_result = StringIO()
    sys.stdout = my_result
    # ---------------------
    scihub_download(paper, paper_type=paper_type, out=out)
    # ---------------------
    sys.stdout = tmp
    h = my_result.getvalue()
    # ---------------------
    # -------
    etat = 0
    if '100%' in h:
        etat = 1
    return etat

def scrape_paper_link():
    DF_DOI = pd.read_excel(PATH + 'list_data_download.xlsx')
    for i in range(0, 1000, 10):
        sleep(random.uniform(0, 1))
        # get url for the each page
        url = "https://scholar.google.com/scholar?start={}&q=Epoxy+amine,+Tg,+Storage+modulus&hl=fr&as_sdt=0,5".format(i) # The retrieval of this URL is presented in the paper in Figure 6: Request link (a) from Google Scholar that will be process with web scraping
        # function for the get content of each page
        doc = get_paperinfo(url)
        # function for the collecting tags
        try:
            paper_tag, link_tag= get_tags(doc)
            # year , author , publication of the paper
            ### year, publication, author = get_author_year_publi_info(author_tag)
            # print(year , publication , author)
            # url of the paper
            link = get_link(link_tag)
            # ********************************************

            papername = get_papertitle(paper_tag)
            if papername is None:
                papername = publication
            print(papername)
            for publication_name,link_pub in zip(papername, link):
                paper = publication_name
                paper_type = "title"
                doi = take_doi(link_pub)
                out = (PATH + doi + '.pdf')
                #---------------------
                etat = Download_SCI(paper,paper_type,out)
                new_row = ({'Journal_Title': publication_name, 'link': link_pub, 'DOI': doi, 'state': etat})
                DF_DOI.loc[len(DF_DOI)] = new_row
                DF_DOI.to_excel(PATH+'list_data_download.xlsx', index=True)
                # ------
                sleep(random.uniform(0, 1))
                print(i)
        except:
            print('failed')
            print(i)
scrape_paper_link()

from habanero import Crossref
cr = Crossref()

"""
# Execute only once to create the file 'list_PDF_from_title.xlsx'
DF = pd.DataFrame(columns=['Journal_Title','DOI', 'state'])
DF.to_excel(PATH+'list_PDF_from_title.xlsx')
"""
DF= pd.read_excel(PATH+'list_PDF_from_title.xlsx')
DF_DOI = pd.read_excel(PATH+'list_data_download.xlsx')
def download_pdf_from_title(DF_DOI,DF):
    Disc=DF_DOI['Journal_Title'][1098:]
    print(len(Disc))
    Disc_unique= []
    for ele in Disc:
        if ele not in Disc_unique: Disc_unique.append(ele)
    print(len(Disc_unique))
    for title in Disc_unique:
        try:
            result = cr.works(query = title)
            paper = result['message']['items'][0]['DOI']
            paper_type = "doi"
            doi = paper[paper.find('/')+1:]
            out = (PATH + doi + '.pdf')
            etat = Download_SCI(paper,paper_type,out)
            new_row = ({'Journal_Title': title, 'DOI': doi, 'state': etat})
        except:
            new_row = ({'Journal_Title': title, 'DOI': 'rien', 'state': 0})
        DF.loc[len(DF)] = new_row
        DF.to_excel(PATH+'list_PDF_from_title.xlsx', index=True)
download_pdf_from_title(DF_DOI,DF)

""" list_PDF_from_title.xlsx and PDFs are in /CCASL/3.1 PDF Document Acquisition/Data_Samples """
