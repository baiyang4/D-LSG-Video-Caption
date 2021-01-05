import shutil
import time
from utils.utils import *
from utils.data import get_train_loader, get_eval_loader
# from utils.opt import parse_opt
# import random
import torch
import torch.nn as nn
import numpy as np
from evaluate import evaluate, convert_data_to_coco_scorer_format
# from tensorboard_logger import configure, log_value
# from models.BYModel import CapModel
from models.model import CapGnnModel


class Run:
    def __init__(self, args, vocab, device, train_loader=None, test_loader=None, test_reference=None, is_debug=True, model_path=None):
        # load training data
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.test_reference = test_reference
        # self.train_loader = get_train_loader(args.train_caption_pkl_path, args.feature_h5_path,
        #                                 args.region_feature_h5_path, args.train_batch_size)
        # self.test_loader = get_eval_loader(args.test_range, args.feature_h5_path,
        #                               args.region_feature_h5_path, args.test_batch_size)
        # self.test_reference = convert_data_to_coco_scorer_format(args.test_reference_txt_path)
        # load vocabulary
        vocab_size = len(vocab)
        print(vocab_size)
        print('dropout = ', args.dropout)
        print('batch size = ', args.train_batch_size)
        print('num_obj = ', args.num_obj)
        print('num_proposals = ', args.num_proposals)
        # create model
        self.model = CapGnnModel(args, vocab).to(device)
        if model_path is not None:
            self.model.load_state_dict(torch.load(model_path, map_location='cuda:0'))
        # parameters for training
        self.args = args
        self.lr = args.learning_rate
        self.epoch_num = args.epoch_num
        self.device = device
        self.train_batch_size = args.train_batch_size
        self.test_batch_size = args.test_batch_size
        self.save_per_epoch = args.save_per_epoch
        self.vocab_size = vocab_size
        self.ss_factor = args.ss_factor
        self.result_handler = ResultHandler(self.model, args, is_debug=is_debug)

    def train(self):
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        # lr_steps = [1,2,3,18]
        lr_steps = [1, 4]
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_steps, gamma=0.5)
        total_step = len(self.train_loader)

        loss_count = 0
        count = 0

        saving_schedule = [int(x * total_step / self.save_per_epoch) for x in list(range(1, self.save_per_epoch + 1))]
        # saving_schedule = [20,200,300]
        print('total: ', total_step)
        print('saving_schedule: ', saving_schedule)
        for epoch in range(self.epoch_num):
            start_time = time.time()
            epsilon = max(0.6, self.ss_factor / (self.ss_factor + np.exp(epoch / self.ss_factor)))
            print('Epoch-{0} lr: {1}'.format(epoch, optimizer.param_groups[0]['lr']))

            for i, (frames, regions, spatials, captions, pos_tags, cap_lens, video_ids) in enumerate(self.train_loader, start=1):
                max_len = max(cap_lens)
                if max_len > 26:
                    max_len = 26
                max_len = 26
                frames = frames.to(self.device)
                # batch_size * sentence_len
                captions = captions[:, :max_len]
                targets = captions.to(self.device)
                regions = regions[:,:,:self.args.num_obj,:].to(self.device)
                optimizer.zero_grad()
                outputs, _, _ = self.model(frames, regions, targets, max_len, epsilon)
                tokens = outputs
                bsz = len(captions)

                # remove pad and flatten outputs
                outputs = torch.cat([outputs[j][:cap_lens[j]] for j in range(bsz)], 0)
                outputs = outputs.view(-1, self.vocab_size)

                # remove pad and flatten targets
                targets = torch.cat([targets[j][:cap_lens[j]] for j in range(bsz)], 0)
                targets = targets.view(-1)

                # compute captioning loss
                cap_loss = criterion(outputs, targets)

                loss_count += cap_loss.item()
                cap_loss.backward()
                # clip_gradient(optimizer, self.args.grad_clip)
                optimizer.step()

                if i % 10 == 0 or bsz < self.train_batch_size:
                    loss_count /= 10 if bsz == self.train_batch_size else i % 10
                    print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f, Perplexity: %5.4f' %
                          (epoch, self.epoch_num, i, total_step, loss_count,
                           np.exp(loss_count)))
                    loss_count = 0
                    tokens = tokens.max(2)[1]
                    tokens = tokens.data[0].squeeze()

                    we = self.model.decoder.decode_tokens(tokens)
                    gt = self.model.decoder.decode_tokens(captions[0].squeeze())

                    print('[vid:%d]' % video_ids[0])
                    print('WE: %s\nGT: %s' % (we, gt))

                if i in saving_schedule:
                    blockPrint()
                    start_time_eval = time.time()
                    self.model.eval()
                    beam_list= [5]
                    metrics_list = []
                    results_list = []
                    for beam_size in beam_list:
                        self.model.update_beam_size(beam_size)
                        metrics, results = evaluate(self.model, self.args, self.test_loader, self.test_reference, multi_modal=True)
                        metrics_list.append(metrics)
                        results_list.append(results)
                    end_time_eval = time.time()
                    enablePrint()
                    print('evaluate time: %.3fs' % (end_time_eval - start_time_eval))

                    self.result_handler.update_result(metrics_list, results_list, epoch)
                    self.model.train()

            end_time = time.time()
            scheduler.step()
            self.result_handler.print_results()
            print("*******One epoch time: %.3fs*******\n" % (end_time - start_time))
        self.result_handler.end_round()

    def train_generator(self, optim_G, batch_size):
        pass

    def test(self):
        reference = convert_data_to_coco_scorer_format(self.args.test_reference_txt_path)
        metrics = evaluate(self.model, self.args, self.test_loader, reference)
