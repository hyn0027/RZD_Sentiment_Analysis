import json
import ast
from gensim.models import keyedvectors
from utils.embedding import embedding_single
import zhconv

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

def loadSentimentCorpus(args, filePath, word2vec, encoding="utf8", ):
    datalines = readtextFile(filePath, encoding=encoding)
    data = []
    cnt = 0
    for item in datalines:
        print(cnt)
        cnt += 1
        word = zhconv.convert(item, "zh-cn").split()
        word2vec, embeddings = embedding_single(args, word2vec, word[1:])
        data.append({
            "label": int(word[0]),
            "words": word[1:],
            "embeddings": embeddings,
        })
    return data