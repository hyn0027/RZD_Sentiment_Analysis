from utils.parseArgument import parseArg
from utils.log import *
from utils.file import *
import os
from model.cnn import CNN
from model.rnn import RNN
from model.mlp import MLP
import torch
from torch.optim import Adam, SGD
from torch.nn import CrossEntropyLoss
import random

def count_parameters(model):
    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

def main():
    args = parseArg()
    logger = getLogger(args=args, name="main")
    logger.info(args)
    match args.model:
        case "cnn":
            logger.info("using model cnn")
            model = CNN(args)
        case "rnn":
            logger.info("using model rnn")
            model = RNN(args)
        case "mlp":
            logger.info("using model mlp")
            model = MLP(args)
    logger.info(model)
    logger.info("trainable parameters: %d", count_parameters(model))
    # exit(-1)
    match args.task:
        case "train":
            word2vec = loadWord2Vec(args.word2vec, True)
            trainData = loadSentimentCorpus(args, os.path.join(args.corpus,"train.txt"), word2vec)
            validData = loadSentimentCorpus(args, os.path.join(args.corpus,"validation.txt"), word2vec)
            criterion = CrossEntropyLoss()
            match args.optimizer:
                case "adam":
                    optimizer = Adam(model.parameters(), args.lr)
                case "SGD":
                    optimizer = SGD(model.parameters(), args.lr)
            bestLoss = 100.0
            previousLoss = 100.0
            earlyStoppingCnt = 0
            lastEpoch = 0
            for i in range(args.max_epoch):
                if not args.no_shuffle:
                    random.shuffle(trainData)
                loss = trainAnEpoch(args, model, trainData, validData, criterion, optimizer, i, bestLoss)
                bestLoss = min(bestLoss, loss)
                if loss < previousLoss:
                    earlyStoppingCnt = 0
                else:
                    earlyStoppingCnt += 1
                previousLoss = loss
                lastEpoch = i
                if earlyStoppingCnt == args.early_stop:
                    logger.info("early stopped!")
                    break
            torch.save(
                {
                    'epoch': lastEpoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                },
                os.path.join(args.save_dir, "lastCheckpoint.pt")
            )
        case "eval":
            word2vec = loadWord2Vec(args.word2vec, True)
            testData = loadSentimentCorpus(args, os.path.join(args.corpus,"test.txt"), word2vec)
            checkpoint = torch.load(os.path.join(args.save_dir, "bestCheckpoint.pt"))
            logger.info("epoch %d", checkpoint["epoch"])
            logger.info("successfully read checkpoint from %s", str(os.path.join(args.save_dir, "bestCheckpoint.pt")))
            model.load_state_dict(checkpoint['model_state_dict'])
            logger.info("successfully load checkpoint from %s", str(os.path.join(args.save_dir, "bestCheckpoint.pt")))
            model.eval()
            TP = 0
            FP = 0
            FN = 0
            TN = 0
            for index in range(len(testData)):
                single_output = model(testData[index]["embeddings"].to(device=0), "valid")
                if testData[index]["label"] == 0:
                    TP += int(single_output[0] > single_output[1])
                    FN += int(single_output[0] < single_output[1])
                else:
                    FP += int(single_output[0] > single_output[1])
                    TN += int(single_output[0] < single_output[1])
            precision = TP / (TP + FP)
            recall = TP / (TP + FN)
            logger.info("accuracy = %f, precision = %f, recall = %f, F = %f",
                        (TP + TN) / len(testData),
                        precision, recall, 2 / (1 / precision + 1 / recall))

def trainAnEpoch(args, model, trainData, validData, criterion, optimizer, epochID, bestLoss):
    logger = getLogger(args=args, name="train epoch " + str(epochID))
    logger.info("begin training epoch %d", epochID)
    batchNum = int((len(trainData) - 1)/ args.batch_size) + 1
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
    # torch.save(
    #     {
    #         'epoch': epochID,
    #         'model_state_dict': model.state_dict(),
    #         'optimizer_state_dict': optimizer.state_dict(),
    #         'loss': loss,
    #     },
    #     os.path.join(args.save_dir, "epoch" + str(epochID) + ".pt")
    # )
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
    return loss

if __name__ == '__main__':
    main()