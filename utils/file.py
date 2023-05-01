import json
import ast
from gensim.models import keyedvectors
from utils.embedding import embedding_single
import zhconv
from utils.log import *
from multiprocessing import Pool, Manager, cpu_count
import random

def readtextFile(filePath, encoding="utf8"):
    try:
        with open(filePath, mode='r', encoding=encoding) as f:
            datas = f.readlines()
            return datas
    except:
        return []

def deleteFile(filePath):
    import os
    if os.path.isfile(filePath):
        os.remove(filePath)
        return True
    return False

def readJsonStringsFile(filePath, encoding="utf8"):
    try:
        with open(filePath, mode='r', encoding=encoding) as f:
            datas = f.readlines()
            dataSet = []
            for item in datas:
                try:
                    dataSet.append(json.loads(item))
                except:
                    try:
                        dataSet.append(ast.literal_eval(item))
                    except:
                        return False
            return dataSet
    except:
        return dict()

def readJsonFile(filePath, encoding="utf8"):
    try:
        with open(filePath, mode='r', encoding=encoding) as f:
            return json.load(f)
    except:
        return {}
    
def writeJsonFile(filePath, content, encoding="utf8"):
    jsonObject = json.dumps(content, indent=4)
    with open(filePath, mode='w', encoding=encoding) as f:
        f.write(jsonObject)

def writeTextFile(filePath, content, encoding="utf8"):
    with open(filePath, mode='w', encoding=encoding) as f:
        for item in content:
            f.write(item + '\n')

def loadWord2Vec(filePath, binary=True):
    return keyedvectors.load_word2vec_format(filePath, binary=binary)

def loadSentimentCorpus(args, filePath, word2vec, encoding="utf8"):
    datalines = readtextFile(filePath, encoding=encoding)
    length = len(datalines)
    logger = getLogger(args=args, name="loadSentimentCorpus")
    logger.info("successfully load %d samples from %s", length, filePath)

    processNum = min(args.max_process, cpu_count())
    processSize = int(length / processNum) + 1
    processArgs = []
    for i in range(int(length / processSize) + 1):
        l = i * processSize
        r = min(length, l + processSize)
        processArgs.append((args, datalines[l:r], word2vec, i))
    data = []
    with Pool(processes=processNum) as pool:
        dataList = pool.starmap(getSentimentCorpusEmbedding, processArgs)
        for item in dataList:
            data += item
    logger.info("successfully processed %d samples from %s", length, filePath)
    return data

def getSentimentCorpusEmbedding(args, datalines, word2vec, processID):
    data = []
    cnt = 0
    logger = getLogger(args=args, name="getSentimentCorpusEmbedding")
    for item in datalines:
        word = zhconv.convert(item, "zh-cn").split()
        embeddings = embedding_single(args, word2vec, word[1:])
        data.append({
            "label": int(word[0]),
            "words": word[1:],
            "embeddings": embeddings
        })
        cnt += 1
        if cnt % args.loading_logging_interval == 0:
            logger.info("finished %d / %d sentences in process %d", cnt, len(datalines), processID)
    return data