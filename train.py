import os
from run import Run
from utils.opt import parse_opt
import torch
import numpy as np
import random
import pickle
from utils.data import get_train_loader, get_eval_loader
from evaluate import convert_data_to_coco_scorer_format
from utils.utils import Vocabulary


if __name__ == "__main__":
    print('----------------------------')
    args = parse_opt()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(args.msrvtt_video_root)
    print('end')

    with open(args.vocab_pkl_path, 'rb') as f:
        vocab = pickle.load(f)

    train_loader = get_train_loader(args.train_caption_pkl_path, args.feature_h5_path,
                                         args.region_feature_h5_path, args.train_batch_size)
    test_loader = get_eval_loader(args.test_range, args.feature_h5_path,
                                       args.region_feature_h5_path, args.test_batch_size)
    test_reference = convert_data_to_coco_scorer_format(args.test_reference_txt_path)

    dropout_list = [0.3]
    glove_list = [True]
    for dropout_i in dropout_list:
        for glove_j in glove_list:
            args.use_glove = glove_j
            args.dropout = dropout_i
            torch.manual_seed(12)
            np.random.seed(12)
            random.seed(12)
            run = Run(args, vocab, device, train_loader=train_loader, test_loader=test_loader,
                      test_reference=test_reference, is_debug=True)
            run.train()


