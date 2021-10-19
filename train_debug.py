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
import torch.distributed as dist
from utils.utils import Vocabulary


if __name__ == "__main__":
    args = parse_opt()
    if args.local_rank < 0:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(args.local_rank)
        device = torch.device(f'cuda:{args.local_rank}')
    multi_gpu = False if args.local_rank < 0 else True
    print(f'multi-gpu = {multi_gpu}, local_rank = {args.local_rank}')
    with open(args.vocab_pkl_path, 'rb') as f:
        vocab = pickle.load(f)

    train_loader, train_sampler = get_train_loader(args.train_caption_pkl_path, args.feature_h5_path,
                                         args.region_feature_h5_path, args.train_batch_size, multi_gpu=multi_gpu)
    test_loader = get_eval_loader(args.test_range, args.feature_h5_path,
                                       args.region_feature_h5_path, args.test_batch_size, multi_gpu=multi_gpu)
    test_reference = convert_data_to_coco_scorer_format(args.test_reference_txt_path)

    seed = 12
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.manual_seed(seed)
    if multi_gpu:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
        # torch.backends.cudnn.benchmark = False
        # torch.backends.cudnn.deterministic = True

    # # args.use_visual_gan = True
    # args.use_lang_gan = True
    # model_path = f'./models_saved/{args.dataset}/{args.dropout}_True/temp'
    # model_name = os.listdir(model_path)[0]
    # model_path = f'{model_path}/{model_name}'
    run = RunGAN(args, vocab, device, train_loader=train_loader, test_loader=test_loader,
                 test_reference=test_reference, is_debug=True, multi_gpu=multi_gpu, train_sampler=train_sampler)
    with torch.backends.cudnn.flags(enabled=False):
        run.train()



