from utils.parseArgument import parseArg
from utils.log import *
from utils.file import *
import os
from cnn import CNN
import torch
import torch.nn as nn
import torch.optim as optim

def main():
    args = parseArg()
    logger = getLogger(args=args, name="main")
    logger.info(args)
    match args.model:
        case "cnn":
            logger.info("using model cnn")
            model = CNN(args)
            logger.info(model)
        case "rnn":
            logger.info("using model rnn")
        case "mlp":
            logger.info("using model mlp")
    match args.task:
        case "train":
            word2vec = loadWord2Vec(args.word2vec, True)
            trainData = loadSentimentCorpus(args, os.path.join(args.corpus,"train.txt"), word2vec)
            validData = loadSentimentCorpus(args, os.path.join(args.corpus,"validation.txt"), word2vec)
            criterion = nn.MSELoss()
            optimizer = optim.SGD(model.parameters(), lr=0.01)
            bestLoss = 100.0
            for i in range(args.max_epoch):
                random.shuffle(trainData)
                bestLoss = trainAnEpoch(args, model, trainData, validData, criterion, optimizer, i, bestLoss)
        case "infer":
            word2vec = loadWord2Vec(args.word2vec, True)
            testData = loadSentimentCorpus(args, os.path.join(args.corpus,"test.txt"), word2vec)
        case "evaluate":
            pass

def trainAnEpoch(args, model, trainData, validData, criterion, optimizer, epochID, bestLoss):
    logger = getLogger(args=args, name="train epoch " + str(epochID))
    logger.info("begin training epoch %d", epochID)
    batchNum = int((len(trainData) - 1)/ args.batch_size) + 1
    # log = getLogger(args, "trainTqdm", False)
    # log.addHandler(TqdmLoggingHandler())
    # for i in tqdm.tqdm(range(batchNum)):
    for i in range(batchNum):
        l = i * args.batch_size
        r = min(l + args.batch_size, len(trainData))
        target = []
        output = []
        optimizer.zero_grad()
        accurateNum = 0
        for index in range(l, r):
            single_output = model(trainData[index]["embeddings"].to(device=0), "train")
            output.append(single_output)
            if trainData[index]["label"] == 0:
                target.append(torch.tensor([1.0, 0.0], device=0))
                accurateNum += int(single_output[0] > single_output[1])
            else:
                target.append(torch.tensor([0.0, 1.0], device=0))
                accurateNum += int(single_output[0] < single_output[1])
        target = torch.stack(target)
        output = torch.stack(output)
        loss = criterion(output, target)
        if (i + 1) % args.logging_interval == 0:
            logger.info("%d / %d batches, loss = %f, accuracy = %f", (i + 1), batchNum, loss, accurateNum / (r - l))
        loss.backward()
        optimizer.step()
    accurateNum = 0
    target = []
    output = []
    for index in range(len(validData)):
        single_output = model(validData[index]["embeddings"].to(device=0), "valid")
        output.append(single_output)
        if validData[index]["label"] == 0:
            target.append(torch.tensor([1.0, 0.0], device=0))
            accurateNum += int(single_output[0] > single_output[1])
        else:
            target.append(torch.tensor([0.0, 1.0], device=0))
            accurateNum += int(single_output[0] < single_output[1])
    target = torch.stack(target)
    output = torch.stack(output)
    loss = criterion(output, target)
    logger.info("validation result: loss = %f, accuracy = %f = %d / %d", loss, accurateNum / len(validData), accurateNum, len(validData))
    torch.save(
        {
            'epoch': epochID,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        },
        os.path.join(args.save_dir, "epoch" + str(epochID) + ":loss=" + str(loss) + ".pt")
    )
    if loss < bestLoss:
        torch.save(
            {
                'epoch': epochID,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            },
            os.path.join(args.save_dir, "bestCheckpoint.pt")
        )
    return min(loss, bestLoss)

if __name__ == '__main__':
    main()