import argparse

def parseArg():
    parser = argparse.ArgumentParser()
    addArg(parser=parser)
    args=parser.parse_args()
    return args

def addArg(parser):
    parser.add_argument(
        "--model", choices=["cnn", "rnn", "mlp"], required=True, 
        help="Selected from: [\"cnn\", \"rnn\", \"mlp\"]"
    )
    parser.add_argument(
        "--verbose", choices=["DEBUG", "INFO", "WARN"], 
        default="INFO", help="Selected from: [debug, info, warn]"
    )
    parser.add_argument(
        "--word2vec", default="../Dataset/wiki_word2vec_50.bin",
        help="path to wiki_word2vec_50.bin"
    )
    parser.add_argument(
        "--corpus", default="../Dataset",
        help="path to corpus data"
    )
    parser.add_argument(
        "--task", choices=["clean", "train", "eval"], required=True, \
        help="Selected from: [clean, train, eval]"
    )
    parser.add_argument(
        "--kdim", default=50, type=int,
        help="k dimension, the dimension for word embeddings"
    )
    parser.add_argument(
        "--max-process", type=int, default=16,
        help="Maximum number of processes"
    )
    parser.add_argument(
        "--loading_logging-interval", type=int, default=200,
        help="logging interval for loading corpus"
    )
    parser.add_argument(
        "--dropout", type=float, default=0.5,
        help="dropout rate"
    )
    parser.add_argument(
        "--batch-size", type=int, default=50,
        help="batch size"
    )
    parser.add_argument(
        "--max-epoch", type=int, default=40,
        help="batch size"
    )
    parser.add_argument(
        "--logging-interval", type=int, default=100,
        help="logging interval for training"
    )
    parser.add_argument(
        "--save-dir", default="./checkpoints/",
        help="checkpoint directory"
    )
    parser.add_argument(
        "--log-file", default="./logging/log.txt",
        help="logging file"
    )
    parser.add_argument(
        "--no-shuffle", action='store_true',
        help="whether to shuffle or not"
    )
    parser.add_argument(
        "--optimizer", default="adam", choices=["adam", "SGD"],
        help="which optimizer to use"
    )
    parser.add_argument(
        "--lr", default=1e-3, type=float,
        help="learning rate"
    )
    parser.add_argument(
        "--early-stop", default=5, type=int,
        help="early stopping epoch"
    )