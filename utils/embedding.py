import torch
import copy
from utils.log import *

def embedding_single(args, word2vec, tokens):
    logger = getLogger(args=args, name="embedding")
    ret = []
    for item in tokens:
        try:
            ret.append(
                torch.from_numpy(copy.deepcopy(word2vec[item]))
            )
        except:
            try:
                for chatacter in item:
                    ret.append(
                        torch.from_numpy(copy.deepcopy(word2vec[chatacter]))
                    )
            except:
                logger.debug("word " + item + " does not exist")
    return torch.stack(ret) if len(ret) > 0 else torch.tensor([])