# Tfidf Clustering Implmenation
This is a quick example of using tf-idf and cosine similirty to build a information retreival system.

# Implementation
## Data preprocessing
Word tokenization, Lowercase all words, remove all stopwords.
## 


# Setup
1. Clone this repository
2. Please Create 2 directory name, 'data' and 'pickle' wihtin this repository
3. Make sure the following packages are install:
   - nltk
   - numpy
   - scipy
   - matplotlib
4. Open up a terminal and run python system.py

# Commands
The system has 3 possible commands:
- python system.py -d: document processing, calculating tfidf score for all documents
- python system.py -q: query processing, allow user to select a model and query the system
- python system.py -c: clustering, clusering using the tfidf scores and display the dendrogram. hierarchy clustering is implemented