import shutil
import time
from utils.utils import *
from utils.data import get_train_loader, get_eval_loader
# from utils.opt import parse_opt
# import random
import torch
import torch.nn as nn
import numpy as np
from evaluate import evaluate, convert_data_to_coco_scorer_format, gather_results, evaluate_multi_gpu
# from tensorboard_logger import configure, log_value
# from models.BYModel import CapModel
from models.model import CapGnnModel, CapBaselineModel, CapBaseline1
from torch import distributed
import collections


class Run:
    def __init__(self, args, vocab, device, train_loader=None, test_loader=None, test_reference=None, is_debug=True, model_path=None, multi_gpu=False, train_sampler=None):
        # load training data
        self.train_loader = train_loader
        self.train_sampler = train_sampler
        self.test_loader = test_loader
        self.test_reference = test_reference
        self.dataset = args.dataset
        print('local_rank----------------', args.local_rank)
        self.local_rank = args.local_rank
        self.multi_gpu = multi_gpu
        # load vocabulary
        if args.dataset == 'msvd':
            args.decode_hidden_size = 1024
        else:
            args.decode_hidden_size = 1300
        vocab_size = len(vocab)
        if self.local_rank <= 0:
            print(vocab_size)
            print('dropout = ', args.dropout)
            print('batch size = ', args.train_batch_size)
            print('decode_hidden_size = ', args.decode_hidden_size)
        self.base_name = self.get_trainer_name(args)
        # create model
        model = CapBaseline1(args, vocab).to(device)
        if self.multi_gpu:
            self.model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                                   find_unused_parameters=True)
        else:
            self.model = model
        # self.model.to(device)
        # self.model = CapBaselineModel(args, vocab).to(device)
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
        self.result_handler = ResultHandler(self.model, self.base_name, is_debug=is_debug, local_rank=self.local_rank)

    def train(self):
        # scaler = torch.cuda.amp.GradScaler()
        criterion = nn.CrossEntropyLoss()
        # optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, betas=(0.5, 0.9))
        # lr_steps = [1,2,3,18]
        lr_steps = [1, 4]
        # if self.dataset == 'msr-vtt':
        #     lr_steps = [1,4,6]
        #     print('lr step msr-vtt = ', lr_steps)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_steps, gamma=0.5)
        total_step = len(self.train_loader)

        loss_count = 0
        loss_count_reduce = 0
        count = 0

        saving_schedule = [int(x * total_step / self.save_per_epoch) for x in list(range(1, self.save_per_epoch + 1))]
        # saving_schedule = [20,200,300]
        print('total: ', total_step)
        print('saving_schedule: ', saving_schedule)
        for epoch in range(self.epoch_num):
            start_time = time.time()
            epsilon = max(0.6, self.ss_factor / (self.ss_factor + np.exp(epoch / self.ss_factor)))

            print('Epoch-{0} lr: {1}'.format(epoch, optimizer.param_groups[0]['lr']))
            if self.dataset == 'msr-vtt' and epoch > 8:
                saving_schedule = [int(x * total_step / 12) for x in
                                   list(range(1, 12 + 1))]
            if self.multi_gpu:
                self.train_sampler.set_epoch(epoch)
            for i, (frames, regions, spatials, captions, pos_tags, cap_lens, video_ids) in enumerate(self.train_loader, start=1):

                if self.dataset == 'msr-vtt':
                    lambda_e = 1 if i < total_step/2 else 2
                    epsilon = max(0.6, self.ss_factor / (self.ss_factor + np.exp((epoch * lambda_e) / self.ss_factor)))

                max_len = 26
                frames = frames.to(self.device)
                # batch_size * sentence_len
                captions = captions[:, :max_len]
                targets = captions.to(self.device)
                regions = regions[:,:,:self.args.num_obj,:].to(self.device)
                optimizer.zero_grad()
                # with torch.cuda.amp.autocast():
                outputs, _, _, _ = self.model(frames, regions, targets, max_len, epsilon)
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

                if self.multi_gpu:
                    # print(type(cap_loss.data))
                    reduced_loss = self.reduce_tensor(cap_loss.data)
                    loss_count_reduce += reduced_loss.item()
                loss_count += cap_loss.item()

                # scaler.scale(cap_loss).backward()
                cap_loss.backward()
                # clip_gradient(optimizer, self.args.grad_clip)
                # scaler.step(optimizer)
                optimizer.step()
                # scaler.update()
                if i % 10 == 0:
                # if i % 10 == 0 or bsz < self.train_batch_size:
                    loss_count /= 10 if bsz == self.train_batch_size else i % 10
                    loss_count_reduce /= 10 if bsz == self.train_batch_size else i % 10
                    print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f, Perplexity: %5.4f' %
                          (epoch, self.epoch_num, i, total_step, loss_count,
                           np.exp(loss_count)))
                    if self.local_rank == 0:
                        print('--------super loss count = ', loss_count_reduce)

                    loss_count = 0
                    loss_count_reduce = 0
                    tokens = tokens.max(2)[1]
                    tokens = tokens.data[0].squeeze()
                    if self.multi_gpu:
                        we = self.model.module.decoder.decode_tokens(tokens)
                        gt = self.model.module.decoder.decode_tokens(captions[0].squeeze())
                    else:
                        we = self.model.decoder.decode_tokens(tokens)
                        gt = self.model.decoder.decode_tokens(captions[0].squeeze())

                    print('[vid:%d]' % video_ids[0])
                    print('WE: %s\nGT: %s' % (we, gt))

                if i in saving_schedule:

                    start_time_eval = time.time()
                    self.model.eval()
                    metrics_list = []
                    results_list = []

                    if self.multi_gpu:
                        results = gather_results(self.model, self.args, self.test_loader, multi_modal=True, multi_gpu=self.multi_gpu)
                        results_multi = [None for _ in range(4)]
                        distributed.all_gather_object(results_multi, results)
                        if self.local_rank == 0:
                            num_keys = [len(results.keys()) for results in results_multi]
                            print('num_keys_sum = ', sum(num_keys))
                            results_all = {**results_multi[0],**results_multi[1],**results_multi[2],**results_multi[3]}
                            print('num_keys_merge = ', len(results_all.keys()))
                            blockPrint()
                            results_all = collections.OrderedDict(sorted(results_all.items()))
                            metrics, results, i_time = evaluate_multi_gpu(results_all, self.test_reference)

                            enablePrint()
                    else:
                        blockPrint()
                        metrics, results, _, i_time = evaluate(self.model, self.args, self.test_loader, self.test_reference, multi_modal=True, multi_gpu=self.multi_gpu)
                        enablePrint()
                    if self.local_rank <= 0:
                        print(len(results.keys()))
                        metrics_list.append(metrics)
                        results_list.append(results)
                        end_time_eval = time.time()
                        enablePrint()
                        print('evaluate time: %.3fs' % (end_time_eval - start_time_eval))
                        print('evaluate inference time: %.3fs' % i_time)
                        self.result_handler.update_result(metrics_list, results_list, epoch)
                    self.model.train()

            end_time = time.time()
            scheduler.step()
            self.result_handler.print_results()
            print("*******One epoch time: %.3fs*******\n" % (end_time - start_time))
        self.result_handler.end_round()

    def train_generator(self, optim_G, batch_size):
        pass

    def get_trainer_name(self, args):
        self.ss_factor = args.ss_factor
        base_name = f'{args.dataset}_{args.ss_factor}_GNN'
        base_name += f'_{args.num_obj}_{args.num_proposals}'
        print(base_name)
        return base_name

    def reduce_tensor(self, tensor):
        rt = tensor.clone()
        distributed.all_reduce(rt, op=distributed.ReduceOp.SUM)
        rt /= distributed.get_world_size()
        return rt

    def test(self):
        reference = convert_data_to_coco_scorer_format(self.args.test_reference_txt_path)
        metrics = evaluate(self.model, self.args, self.test_loader, reference)
