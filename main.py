from utils.parseArgument import parseArg
from utils.log import *
from utils.file import *
import os
from cnn import CNN

def main():
    args = parseArg()
    logger = getLogger(args=args, name="main")
    logger.info(args)
    match args.model:
        case "cnn":
            logger.info("using model cnn")
            model = CNN(args)
            print(model)
        case "rnn":
            logger.info("using model rnn")
        case "mlp":
            logger.info("using model mlp")
    match args.task:
        case "train":
            word2vec = loadWord2Vec(args.word2vec, True)
            trainData = loadSentimentCorpus(args, os.path.join(args.corpus,"train.txt"), word2vec)
            validData = loadSentimentCorpus(args, os.path.join(args.corpus,"validation.txt"), word2vec)
            print(trainData[0])
        case "infer":
            word2vec = loadWord2Vec(args.word2vec, True)
            testData = loadSentimentCorpus(args, os.path.join(args.corpus,"test.txt"), word2vec)
        case "evaluate":
            pass

if __name__ == '__main__':
    main()