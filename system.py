import os
import sys
import nltk
import string
import math
import pickle
import numpy as np

from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib import pyplot as plt
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

#=========================================================================================================================================
#Methods
#Remove stopwords, lowercase all character, stemming, remove numbers and tokenize all words, remove puntuations
#Collect term frequency and inverse document frequnecy
def documentProcessing(fileData,currentTf,fileName,option):
    stop_words = set(stopwords.words('english'))
    ps = PorterStemmer()
    if option == "-tfidf":
        for eachLine in fileData:
            tokenizeLine = nltk.tokenize.word_tokenize(eachLine)
            for eachWord in tokenizeLine:
                eachWord = eachWord.lower()
                eachWord = eachWord.translate(str.maketrans('', '', string.punctuation))
                if not eachWord in stop_words and not eachWord.isnumeric() and not len(eachWord) == 1 and not len(eachWord) == 0:
                    stemWord = ps.stem(eachWord)
                    if not stemWord in currentTf:
                        currentTf[stemWord] = 1
                    else:
                        currentTf[stemWord] += 1
        returnObject = {'fileName':fileName,'allTerms':currentTf}
        return returnObject
    elif option=="-brown":
        text = ""
        for eachLine in fileData:
            tokenizeLine = nltk.tokenize.word_tokenize(eachLine)
            for eachWord in tokenizeLine:
                eachWord = eachWord.lower()
                eachWord = eachWord.translate(str.maketrans('', '', string.punctuation))
                if not eachWord in stop_words and not eachWord.isnumeric() and not len(eachWord) == 1 and not len(eachWord) == 0:
                    stemWord = ps.stem(eachWord)
                    text += stemWord + " "
        returnObject = {'fileName':fileName,'text':text}
        return returnObject

#Calculate word appear in document through entire document collection
def wordAppearInDocCal(termApperInDoc,currentDocTerms):
    for eachKey in currentDocTerms:
        if not eachKey in termApperInDoc:
            termApperInDoc[eachKey] = 1
        else:
            termApperInDoc[eachKey] += 1
    return termApperInDoc

#Calculate tf-idf score for each words within each document
def tfidfCal(allTf,termApperInDoc,totalDocNum):
    tfidfList = []
    for eachDocument in allTf:
        allTerms = eachDocument['allTerms']
        wordCounts = len(allTerms.keys())
        curentDocTfidf = {}
        for eachWord in allTerms:
            tf = allTerms[eachWord]/wordCounts
            idf = math.log(totalDocNum/termApperInDoc[eachWord]+1)+1
            curentDocTfidf[eachWord] = tf*idf
        tfidf={'fileName':eachDocument['fileName'],'tfidfScores':curentDocTfidf}
        tfidfList.append(tfidf)
    return tfidfList

#Load documents pickle files
def loadDocumentPickles(pickleFilePath):
    documents = []
    listOfFiles = os.listdir(pickleFilePath)
    for eachFile in listOfFiles:
        currentFilePath = os.path.join(pickleFilePath,eachFile)
        currentFile = open(currentFilePath,'rb')
        doucmentData = pickle.load(currentFile)
        documents.append(doucmentData)
        currentFile.close()
    return documents

#Processing Query
def queryProcessing(userQuery):
    stop_words = set(stopwords.words('english'))
    ps = PorterStemmer()
    tokenizeQuery = nltk.tokenize.word_tokenize(userQuery)
    queryTf = {}
    for eachWord in tokenizeQuery:
        if not eachWord in stop_words and not eachWord.isnumeric() and not len(eachWord) == 1 and not len(eachWord) == 0:
            stemWord = ps.stem(eachWord)
            if not stemWord in queryTf:
                queryTf[stemWord] = 1
            else:
                queryTf[stemWord] += 1
    return queryTf

#Regular scoring by adding terms and score
def addScoreSimilirity(queryTf,documents):
    tfidfScoreLists = []
    for eachDoc in documents:
        tfidfScores = eachDoc['tfidfScores']
        eachDocScore = 0
        for eachWord in queryTf:
            counts = queryTf[eachWord]
            if eachWord in tfidfScores:
                eachDocScore += counts*tfidfScores[eachWord]
        tfidfScoreLists.append({'fileName':eachDoc['fileName'],'finalScore':eachDocScore})
    return tfidfScoreLists

#Cosine similirty Scoring
def cosineSimilirity(queryTf,documents):
    tfidfScoreLists= []
    for eachDoc in documents:
        tfidfScore = eachDoc['tfidfScores']
        currentTop = 0
        botQuery = 0
        botDoc = 0
        for eachWord in queryTf:
            queryCount = queryTf[eachWord]
            docCount = 0
            if eachWord in tfidfScore:
                docCount = tfidfScore[eachWord]
            currentTop += queryCount*docCount
            botQuery += queryCount**2
            botDoc += docCount **2
        botQuery = math.sqrt(botQuery)
        botDoc = math.sqrt(botDoc)
        currentBot = botQuery*botDoc
        if currentBot == 0:
            cosineSimilirityScore = 0
        else:
            cosineSimilirityScore = currentTop/currentBot
        tfidfScoreLists.append({'fileName':eachDoc['fileName'],'finalScore':cosineSimilirityScore})
    return tfidfScoreLists

#Get all words within document collection
def getAllWords(documents):
    allWordSet = set()
    for eachDoc in documents:
        currentWords = eachDoc['tfidfScores'].keys()
        for eachWord in currentWords:
            allWordSet.add(eachWord)
    return allWordSet

#Generate numpy array for each documents
def documentArrayGeneration(documents,allWordSet):
    allDocsList = []
    for eachDoc in documents:
        tfidfScoreDict = eachDoc['tfidfScores']
        currentDocList = []
        for eachWord in allWordSet:
            if eachWord in tfidfScoreDict:
                currentDocList.append(tfidfScoreDict[eachWord])
            else:
                currentDocList.append(0)
        allDocsList.append(currentDocList)
    return allDocsList

#Pretty print output
def prettyPrint(inputList):
    for each in inputList:
        print(each)
#=========================================================================================================================================
#Main
cwd = os.getcwd()
dataPath = os.path.join(cwd,"data")
pickleTfidfPath = os.path.join(cwd,"pickleTfidf")
pickleWordsPath = os.path.join(cwd,"pickleWords")
brownClusterPath = os.path.join(cwd,"tan-clustering")
brownCLusterOutputPath = os.path.join(cwd,"brownClusterOutput")
sys.path.append(brownClusterPath)
import pmi_cluster as brownCluster

userCommand = ""
command = sys.argv
if len(command) > 1:
    userCommand = command[1]

#Document processing, data preprocessing and pickle creation
if userCommand == "-d":
    if len(command) > 2:
        docProcessOption = command[2]
    else:
        print("Please provide what method you would like to use. Ex. python system.py -d -tfidf")
        print("-tfdif for tfidf document processing")
        print("-brown for brown clustering document processing")
        sys.exit()
    if docProcessOption == "-tfidf":
        #documents processing, record all tfidf score, and pickle the data
        listOfFiles = os.listdir(dataPath)
        totalDocNum = len(listOfFiles)
        allTf=[]
        termApperInDoc = {}
        for eachFile in listOfFiles:
            print("Processing "+eachFile)
            currentFilePath = os.path.join(dataPath,eachFile)
            currentFile = open(currentFilePath,'r',errors="ignore")
            currentTf = {}
            currentDic = documentProcessing(currentFile,currentTf,eachFile,'-tfidf')
            termApperInDoc= wordAppearInDocCal(termApperInDoc,currentDic['allTerms'])
            allTf.append(currentDic)

        tfidfList = tfidfCal(allTf,termApperInDoc,totalDocNum)
        for eachDocument in tfidfList:
            docName = eachDocument['fileName'].replace(".txt",".pickle")
            pickleFilePath = os.path.join(pickleTfidfPath,docName)
            pickleFile = open(pickleFilePath,'wb')
            pickle.dump(eachDocument,pickleFile)
            pickleFile.close()
    elif docProcessOption == "-brown":
        #documents processing, record all tfidf score, and pickle the data
        listOfFiles = os.listdir(dataPath)
        totalDocNum = len(listOfFiles)
        docList = []
        for eachFile in listOfFiles:
            print("Processing "+eachFile)
            currentFilePath = os.path.join(dataPath,eachFile)
            currentFile = open(currentFilePath,'r',errors="ignore")
            currentTf = {}
            currentDic = documentProcessing(currentFile,currentTf,eachFile,'-brown')
            docList.append(currentDic)
        for eachDocument in docList:
            docName = eachDocument['fileName']
            text = eachDocument['text']
            pickleFilePath = os.path.join(pickleWordsPath,docName)
            pickleFile = open(pickleFilePath,'w')
            pickleFile.write(text)
            #pickle.dump(eachDocument,pickleFile)
            pickleFile.close()

#Query processing, user searching
elif userCommand == "-q":
    documents = loadDocumentPickles(pickleTfidfPath)
    scoreList= []
    #userInput = "deep learning question answering system"
    #userInput = "ibm watson"
    userInput = input('Please provide a query: ')
    queryTf = queryProcessing(userInput)
    print("Please Select a Model:")
    print("1. Raw Count Model")
    print("2. Cosine Similirity Model")
    modelSelection = input()
    #modelSelection = 2
    
    if modelSelection == "1":
        scoreList = addScoreSimilirity(queryTf,documents)
    elif modelSelection == "2":
        scoreList= cosineSimilirity(queryTf,documents)
    else:
        print("Please select the correct model by given number 1 or 2")
        sys.exit()

    sortedList = sorted(scoreList, key=lambda k: k['finalScore'],reverse=True)
    prettyPrint(sortedList)

elif userCommand == "-c":
    if len(command) > 2:
        clusteringOption = command[2]
    else:
        print("Please provide what method you would like to use. Ex. python system.py -c -METHOD")
        print("METHOD: -single, -complete, -average, -weighted, -centroid")
        sys.exit()
    clusterMethod =""
    if clusteringOption == '-single':
        clusterMethod = "single"
    elif clusteringOption == "-complete":
        clusterMethod = "complete"
    elif clusteringOption == "-average":
        clusterMethod = "average"
    elif clusteringOption == "-weighted":
        clusterMethod = "weighted"
    elif clusteringOption == "-centroid":
        clusterMethod = "centroid"
    else:
        print("invalid method provided")
        print("METHOD: -single, -complete, -average, -weighted, -centroid")
        sys.exit()

    clusteringOption = ""
    documents = loadDocumentPickles(pickleTfidfPath)
    documentsCount = len(documents)
    print("Geting All Words")
    wordsSet = getAllWords(documents)
    print("Creating Numpy Array")
    doctfidfList = documentArrayGeneration(documents,wordsSet)
    numpyArray = np.array(doctfidfList)
    print(numpyArray)
    linked = linkage(numpyArray, clusterMethod)
    print(linked)
    plt.figure(figsize=(10, 7))
    dendrogram(linked,
            orientation='top',
            distance_sort='descending',
            show_leaf_counts=True)
    plt.show()

elif userCommand == "-cran":
    cranfieldPath = os.path.join(cwd,"cranfield")
    cranfieldQueriesPath = os.path.join(cranfieldPath,"cranQueries")
    cranfieldQueryResultPath = os.path.join(cranfieldPath,"cranqrel.txt")
    listOfQueries = os.listdir(cranfieldQueriesPath)
    documents = loadDocumentPickles(pickleTfidfPath)

    print("Please Select a Model:")
    print("1. Raw Count Model")
    print("2. Cosine Similirity Model")
    modelSelection = input()
    #modelSelection = "2"
    #go through each query and compute similirty score
    queryResultList = []
    for eachQueryFile in listOfQueries:
        currentQueryFilePath = os.path.join(cranfieldQueriesPath,eachQueryFile)
        currentQueryFile = open(currentQueryFilePath,'r')
        query = ""
        for eachLine in currentQueryFile:
            query+=eachLine
        currentQueryFile.close()
        queryTf = queryProcessing(query.strip())
        if modelSelection == "1":
            scoreList = addScoreSimilirity(queryTf,documents)
        elif modelSelection == "2":
            scoreList= cosineSimilirity(queryTf,documents)
        else:
            print("Please select the correct model by given number 1 or 2")
            sys.exit()
        sortedList = sorted(scoreList, key=lambda k: k['finalScore'],reverse=True)
        queryResultList.append({"queryFileName":eachQueryFile,"scoreList":sortedList})
    #import cranfield query result
    cranfieldResultFile = open(cranfieldQueryResultPath,"r")
    cranfieldResultDict = {}
    for eachline in cranfieldResultFile:
        splitList = eachline.split(' ')
        if not splitList[0] in cranfieldResultDict:
            cranfieldResultDict[splitList[0]] = [splitList[1]]
        else:
            cranfieldResultDict[splitList[0]].append(splitList[1])
    #evoluation
    precisionRecallList = []
    for eachQueryResult in queryResultList:
        queryid = eachQueryResult['queryFileName'].replace(".txt","")
        correctHit = 0
        totalRetreival = 0
        for eachScore in eachQueryResult['scoreList']:
            if not eachScore['finalScore'] == 0:
                totalRetreival+=1
                if eachScore['fileName'].replace(".txt","") in cranfieldResultDict[queryid]:
                    correctHit+=1
        precision = correctHit/totalRetreival
        recall = correctHit/len(cranfieldResultDict[queryid])
        precisionRecallList.append({"queryid":queryid,"precision":precision,"recall":recall})
        prettyPrint(precisionRecallList)
elif userCommand == "-b":
    listOfFiles = os.listdir(pickleWordsPath)
    for eachFile in listOfFiles:
        currentFilePath = os.path.join(pickleWordsPath,eachFile)
        word_counts = brownCluster.make_word_counts(brownCluster.document_generator(currentFilePath, False),None,1)
        c = brownCluster.DocumentLevelClusters(brownCluster.document_generator(currentFilePath, False),word_counts, 1000)
        outpufFilePath = os.path.join(brownCLusterOutputPath,eachFile)
        c.save_clusters(outpufFilePath)
else:
    print("Extra commands require")
    print("-d for document processing")
    print("-q for query processing")
    print("-c for clustering")
    print("-cran for cranfield query evoluations")
    print("-b for brown word clustering")