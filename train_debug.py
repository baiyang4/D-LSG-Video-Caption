import os
from run import Run
from run_gun import RunGAN
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

    torch.manual_seed(12)
    np.random.seed(12)
    random.seed(12)

    # # args.use_visual_gan = True
    # args.use_lang_gan = True
    # model_path = f'./models_saved/{args.dataset}/{args.dropout}_True/temp'
    # model_name = os.listdir(model_path)[0]
    # model_path = f'{model_path}/{model_name}'
    run = RunGAN(args, vocab, device, train_loader=train_loader, test_loader=test_loader,
                 test_reference=test_reference, is_debug=True)
    with torch.backends.cudnn.flags(enabled=False):
        run.train()



