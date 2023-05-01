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
        "--task", choices=["clean", "train", "infer", "evaluate"], required=True, \
        help="Selected from: [clean, train, infer, evaluate]"
    )
    parser.add_argument(
        "--kdim", default=50, type=int,
        help="k dimension, the dimension for word embeddings"
    )
    parser.add_argument(
        "--max-process", type=int, default=20,
        help="Maximum number of processes"
    )
    parser.add_argument(
        "--logging-interval", type=int, default=50,
        help="logging interval for loading corpus"
    )
    parser.add_argument(
        "--dropout", type=float, default=0.5,
        help="dropout rate"
    )