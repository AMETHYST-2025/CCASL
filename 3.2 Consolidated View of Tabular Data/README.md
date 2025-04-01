
# 3.2 Consolidated View of Tabular Data

This is the most important part of the work because it contains the programs for extraction, relevance recommendation, table normalization, and data integration. 
These programs can be executed in the following order
- [ ] `Table_Extraction.py`: Captures all the tables in a PDF.
- [ ] `Recommendation.py`: Classifies images of tables based on their relevance using keywords.
- [ ] `textract.py` : Contains the code for **image preprocessing**, and the connection to the Amazon Textract API, but we recommend using the web interface, which also offers a free trial. [AWSTextract](https://aws.amazon.com/fr/textract/)
- [ ] `Normalization.py` & `Samples_Normalization.py`: A global rule-based program designed to normalize certain frequent attributes (e.g., chemical properties) in epoxy-amine publications. A machine learning-based approach with RNN is located in `RNN_column_name.py`.
- [ ] `ChemDataExtractor.py`: is use in `Normalization.py` to detect acronym and abbreviation, notably 'EP' which is usually describe in "Experimental and method"
- [ ] `Data_Integration.py`: Creates an integrated view of the case study from the paper (V(EA), SM_bf, SM_af, Tg).
- [ ] `Data`: Contains samples of intermediate results, more data is given on [zotero](https://zenodo.org/records/15115892)

