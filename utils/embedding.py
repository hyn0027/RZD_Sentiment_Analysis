import torch
import numpy as np
import copy

def embedding_single(args, word2vec, tokens):
    ret = []
    for item in tokens:
        try:
            ret.append(
                torch.from_numpy(copy.deepcopy(word2vec[item])).to(device=0)
            )
        except:
            try:
                for chatacter in item:
                    ret.append(
                        torch.from_numpy(copy.deepcopy(word2vec[chatacter])).to(device=0)
                    )
            except:
                print("word " + item + " does not exist")
    return word2vec, torch.stack(ret) if len(ret) > 0 else torch.tensor([], device=0)