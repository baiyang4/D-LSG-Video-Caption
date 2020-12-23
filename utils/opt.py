# coding: utf-8
import argparse
import time
import os

def parse_opt():
    # data path
    data_pth = '/mnt/CAC593A17C0101D9/DL_projects/Other_projects/video description/RMN-master/data'
    if not os.path.exists(data_pth):
        data_pth = '/home/mist/video_captioning/data'
    # data_pth = '../video description/RMN-master/data'
    # parser
    parser = argparse.ArgumentParser()
    # General settings
    parser.add_argument('--dataset', type=str, default='msvd', help='choose from msvd|msr-vtt')
    parser.add_argument('--epoch_num', type=int, default=40)
    parser.add_argument('--save_per_epoch', type=int, default=8)
    parser.add_argument('--train_batch_size', type=int, default=128)
    parser.add_argument('--test_batch_size', type=int, default=64)
    parser.add_argument('--beam_size', type=int, default=3)
    parser.add_argument('--use_glove', type=bool, default=False)

    # Network settings
    parser.add_argument('--model', type=str, default='RMN')  # RMN
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--frame_hidden_size', type=int, default=1000)
    parser.add_argument('--motion_hidden_size', type=int, default=1000)
    parser.add_argument('--visual_hidden_size', type=int, default=1024)
    parser.add_argument('--region_projected_size', type=int, default=1000)
    parser.add_argument('--spatial_projected_size', type=int, default=300)
    parser.add_argument('--word_size', type=int, default=300)
    parser.add_argument('--hidden_size', type=int, default=1300)
    parser.add_argument('--att_size', type=int, default=1024)
    parser.add_argument('--time_size', type=int, default=300)
    parser.add_argument('--query_hidden_size', type=int, default=1024)
    parser.add_argument('--decode_hidden_size', type=int, default=1024)
    parser.add_argument('--ss_factor', type=int, default=20)

    # Optimization settings
    parser.add_argument('--learning_rate', type=float, default=0.00008)
    parser.add_argument('--learning_rate_decay', type=int, default=1)
    parser.add_argument('--learning_rate_decay_every', type=int, default=10)
    parser.add_argument('--learning_rate_decay_rate', type=float, default=10)
    parser.add_argument('--grad_clip', type=float, default=10)

    # Feature extract settings
    parser.add_argument('--max_frames', type=int, default=26)
    parser.add_argument('--max_words', type=int, default=26)
    parser.add_argument('--num_boxes', type=int, default=36)
    parser.add_argument('--a_feature_size', type=int, default=1536)
    parser.add_argument('--m_feature_size', type=int, default=1024)
    parser.add_argument('--region_feature_size', type=int, default=2048)
    parser.add_argument('--spatial_feature_size', type=int, default=5)

    # Dataset settings
    parser.add_argument('--msrvtt_video_root', type=str, default=f'{data_pth}/MSR-VTT/Videos/')
    parser.add_argument('--msrvtt_anno_trainval_path', type=str, default=f'{data_pth}/MSR-VTT/train_val_videodatainfo.json')
    parser.add_argument('--msrvtt_anno_test_path', type=str, default=f'{data_pth}/MSR-VTT/test_videodatainfo.json')
    parser.add_argument('--msrvtt_anno_json_path', type=str, default=f'{data_pth}/MSR-VTT/datainfo.json')
    parser.add_argument('--msrvtt_train_range', type=tuple, default=(0, 6513))
    parser.add_argument('--msrvtt_val_range', type=tuple, default=(6513, 7010))
    parser.add_argument('--msrvtt_test_range', type=tuple, default=(7010, 10000))

    parser.add_argument('--msvd_video_root', type=str, default=f'{data_pth}/MSVD/youtube_videos')
    parser.add_argument('--msvd_csv_path', type=str, default=f'{data_pth}/MSVD/video_corpus.csv')
    parser.add_argument('--msvd_video_name2id_map', type=str, default=f'{data_pth}/MSVD/youtube_mapping.txt')
    parser.add_argument('--msvd_anno_json_path', type=str, default=f'{data_pth}/MSVD/annotations.json')
    parser.add_argument('--msvd_train_range', type=tuple, default=(0, 1200))
    parser.add_argument('--msvd_val_range', type=tuple, default=(1200, 1300))
    parser.add_argument('--msvd_test_range', type=tuple, default=(1300, 1970))

    # Result path
    parser.add_argument('--result_dir', type=str, default='./results/msr-vttgumbel')
    args = parser.parse_args()
    if not os.path.exists(args.result_dir):
        os.mkdir(args.result_dir)
    args.val_prediction_txt_path = os.path.join(args.result_dir, args.dataset + '_val_predictions.txt')
    args.val_score_txt_path = os.path.join(args.result_dir, args.dataset + '_val_scores.txt')
    args.test_prediction_txt_path = os.path.join(args.result_dir, args.dataset + '_test_predictions.txt')
    args.test_score_txt_path = os.path.join(args.result_dir, args.dataset + '_test_scores.txt')

    args.model_pth_path = os.path.join(args.result_dir, args.dataset + '_model.pth')
    args.best_meteor_pth_path = os.path.join(args.result_dir, args.dataset + '_best_meteor.pth')
    args.best_cider_pth_path = os.path.join(args.result_dir, args.dataset + '_best_cider.pth')
    args.optimizer_pth_path = os.path.join(args.result_dir, args.dataset + '_optimizer.pth')
    args.best_meteor_optimizer_pth_path = os.path.join(args.result_dir, args.dataset + '_best_meteor_optimizer.pth')
    args.best_cider_optimizer_pth_path = os.path.join(args.result_dir, args.dataset + '_best_cider_optimizer.pth')

    # caption and visual features path
    # args.dataset = 'msvd'
    if args.dataset == 'msvd':
        args.feat_dir = f'{data_pth}/MSVD'
    elif args.dataset == 'msr-vtt':
        args.feat_dir = f'{data_pth}/MSR-VTT'
    else:
        raise ValueError('choose one dataset from msvd|msr-vtt')
    args.val_reference_txt_path = os.path.join(args.feat_dir, args.dataset + '_val_references.txt')
    args.test_reference_txt_path = os.path.join(args.feat_dir, args.dataset + '_test_references.txt')
    args.vocab_pkl_path = os.path.join(args.feat_dir, args.dataset + '_vocab.pkl')
    args.caption_pkl_path = os.path.join(args.feat_dir, args.dataset + '_captions.pkl')
    caption_pkl_base = os.path.join(args.feat_dir, args.dataset + '_captions')
    args.train_caption_pkl_path = caption_pkl_base + '_train.pkl'
    args.val_caption_pkl_path = caption_pkl_base + '_val.pkl'
    args.test_caption_pkl_path = caption_pkl_base + '_test.pkl'

    args.feature_h5_path = os.path.join(args.feat_dir, args.dataset + '_features.h5')
    args.feature_h5_feats = 'feats'
    args.feature_h5_lens = 'lens'

    if args.dataset == 'msvd':
        args.region_feature_h5_path = f'{data_pth}/MSVD/msvd_region_feature.h5'
    elif args.dataset == 'msr-vtt':
        args.region_feature_h5_path = f'{data_pth}/MSR-VTT/msrvtt_region_feature.h5'
    args.region_visual_feats = 'vfeats'
    args.region_spatial_feats = 'sfeats'

    args.video_sort_lambda = lambda x: int(x[5:-4])
    dataset = {
        'msr-vtt': [args.msrvtt_video_root, args.msrvtt_anno_json_path,
                    args.msrvtt_train_range, args.msrvtt_val_range, args.msrvtt_test_range],
        'msvd': [args.msvd_video_root, args.msvd_anno_json_path,
                 args.msvd_train_range, args.msvd_val_range, args.msvd_test_range]
    }
    args.video_root, args.anno_json_path, args.train_range, args.val_range, args.test_range = dataset[args.dataset]

    # tensorboard log path
    time_format = '%m-%d_%X'
    current_time = time.strftime(time_format, time.localtime())
    env_tag = '%s' % (current_time) + args.result_dir
    # args.log_environment = os.path.join('logs', env_tag)
    args.log_environment = 'logs/hehe' # env_tag windows

    return args


if __name__ == '__main__':
    opt = parse_opt()
    print(opt.feat_dir)
    print(opt.msrvtt_train_range)
    print(opt.video_root)
