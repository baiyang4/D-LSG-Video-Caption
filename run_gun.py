
import time
from utils.utils import *

import torch
import torch.nn as nn
import numpy as np
from evaluate import evaluate, convert_data_to_coco_scorer_format, gather_results, evaluate_multi_gpu
from models.model import CapGnnModel, DiscV2
from torch.utils.tensorboard import SummaryWriter
import math
from torch import distributed
import collections
import seaborn as sns
import matplotlib.pyplot as plt
from statistics import median


class RunGAN:
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

        if args.dataset == 'msvd':
            args.decode_hidden_size = 1024
            args.num_proposals = 8
            args.num_obj = 16
            args.num_topk = 3
        else:
            args.decode_hidden_size = 1536
            args.num_proposals = 5
            args.num_obj = 36
            args.num_topk = 5

        self.base_name = self.get_trainer_name(args)
        vocab_size = len(vocab)
        print(vocab_size)

        if self.local_rank <= 0:
            print(vocab_size)
            print('dropout = ', args.dropout)
            print('batch size = ', args.train_batch_size)
            print('decode_hidden_size = ', args.decode_hidden_size)
        # initialize generation model

        # self.checkpoint = torch.load('10.pt', map_location=device)
        self.checkpoint = None
        if self.checkpoint is None:
            self.last_epoch = -1
        else:
            self.last_epoch = self.checkpoint['epoch']
        self.model = CapGnnModel(args, vocab).to(device)
        if self.checkpoint is not None:
            self.model.load_state_dict(self.checkpoint['model_state_dict'])
        if self.multi_gpu:
            self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[args.local_rank],
                                                                   find_unused_parameters=True)
        # initialize discriminative model
        if self.use_visual_gan:
            self.D_visual = DiscV2(args, vocab_size).to(device)
            if self.checkpoint is not None:
                self.D_visual.load_state_dict(self.checkpoint['model_d_state_dict'])
            if self.multi_gpu:
                self.D_visual = torch.nn.parallel.DistributedDataParallel(self.D_visual, device_ids=[args.local_rank],
                                                                          find_unused_parameters=True)
        # load hyper-parameters
        self.args = args
        self.lr = args.learning_rate
        self.epoch_num = args.epoch_num
        self.device = device
        self.train_batch_size = args.train_batch_size
        self.test_batch_size = args.test_batch_size
        self.save_per_epoch = args.save_per_epoch
        self.vocab_size = vocab_size
        self.ss_factor = args.ss_factor
        # initialize results handling tool
        self.result_handler = ResultHandler(self.model, self.base_name, is_debug=is_debug, local_rank=self.local_rank)
        # initialize tensorboard
        if self.local_rank <= 0:
            self.writer = SummaryWriter(comment=self.base_name)

    def train(self):
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, betas=(0.5, 0.9))
        if self.checkpoint is not None:
            optimizer.load_state_dict(self.checkpoint['optimizer_state_dict'])
        lr_steps = [4, 7]
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_steps, gamma=0.5, last_epoch=self.last_epoch)

        total_step = len(self.train_loader)
        if self.use_visual_gan:
            lr_steps_D = [1, 4]
            optimizer_D = torch.optim.Adam(self.D_visual.parameters(), lr=self.lr, betas=(0.5, 0.9))
            if self.checkpoint is not None:
                optimizer_D.load_state_dict(self.checkpoint['optimizer_d_state_dict'])
            scheduler_D = torch.optim.lr_scheduler.MultiStepLR(optimizer_D, milestones=lr_steps_D,
                                                               gamma=0.5, last_epoch=self.last_epoch)
            # decrease Discriminator loss weights dynamically
            if self.checkpoint is not None:
                gan_lambda_handler = GANLambdaHandler(total_step, self.lambda_D_visual, cap_list=self.checkpoint['cap_list'])
            else:
                gan_lambda_handler = GANLambdaHandler(total_step, self.lambda_D_visual)
        loss_count = 0
        loss_count_G = 0
        loss_count_D = 0
        wasserstein = 0

        saving_schedule_small = [int(x * total_step / 2) for x in list(range(1, 2 + 1))]
        saving_schedule_mid = [int(x * total_step / 8) for x in list(range(1, 8 + 1))]
        saving_schedule_high = [int(x * total_step / 12) for x in list(range(1, 12 + 1))]
        if self.local_rank <= 0:
            if not os.path.exists(f'./images/{self.base_name}'):
                os.makedirs(f'./images/{self.base_name}')
        print('total: ', total_step)
        for epoch in range(self.last_epoch + 1, self.epoch_num):

            if epoch < 4:
                saving_schedule = saving_schedule_small
            elif epoch < 7:
                saving_schedule = saving_schedule_mid
            else:
                if self.dataset =='msr-vtt':
                    print('msr-vtt')
                    saving_schedule = saving_schedule_high
                else:
                    saving_schedule = saving_schedule_mid

            start_time = time.time()
            epsilon = max(0.6, self.ss_factor / (self.ss_factor + np.exp(epoch / self.ss_factor)))
            if self.local_rank <= 0:
                print('Epoch-{0} lr: {1}'.format(epoch, optimizer.param_groups[0]['lr']))
            if self.use_visual_gan:
                if self.local_rank <= 0:
                    print('Epoch-{0} lr visual GAN: {1}'.format(epoch, optimizer_D.param_groups[0]['lr']))
                    # print('visual gan lambda: ', visual_gan_lambda[epoch])

            if self.multi_gpu and self.local_rank<=0:
                self.train_loader.sampler.set_epoch(epoch)

            for i, (frames, regions, spatials, captions, pos_tags, cap_lens, video_ids) in enumerate(self.train_loader, start=1):

                if self.dataset == 'msr-vtt':
                    lambda_e = 1 if i < total_step/2 else 2
                    epsilon = max(0.6, self.ss_factor / (self.ss_factor + np.exp((epoch * 2 + lambda_e) / self.ss_factor)))

                max_len = 26
                bs = frames.shape[0]

                frames = frames.to(self.device)
                # batch_size * sentence_len
                regions = regions[:, :, :self.args.num_obj, :].to(self.device)
                captions = captions[:, :max_len]
                targets = captions.to(self.device)

                """ Train D """
                if self.use_visual_gan:
                    seq_mask = (captions > 0).to(torch.float32)
                    att_mask = torch.matmul(seq_mask.view(bs, max_len, 1), seq_mask.view(bs, 1, max_len)).to(
                        self.device)
                    f_caption, object_psl, motion_psl, alpha_all = self.model(frames, regions, targets, max_len, epsilon)

                if self.use_visual_gan:
                    f_caption = f_caption.detach()
                    r_caption = self.to_onehot(targets, self.vocab_size)
                    object_psl = object_psl.detach()
                    motion_psl = motion_psl.detach()
                    alpha_all = alpha_all.detach()
                    loss_count_D, wasserstein = \
                        self.train_disc(r_caption, f_caption, optimizer_D, self.D_visual, self.num_D_visual,
                                        i, epoch, total_step, loss_count_D, wasserstein, obj_psl=object_psl,
                                        motion_psl=motion_psl, pos_tag=pos_tags, att_mask=att_mask, alpha_all=alpha_all)

                """ Train Captioning Model """
                optimizer.zero_grad()

                outputs, object_psl, motion_psl, alpha_all = self.model(frames, regions, targets, max_len, epsilon)

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
                total_loss = cap_loss

                cap_loss_record = cap_loss.item()

                if self.multi_gpu:
                    cap_loss_record = self.reduce_tensor(cap_loss.data).item()

                loss_count += cap_loss_record

                if self.local_rank <= 0:
                    self.writer.add_scalar('Loss/cap_loss', cap_loss_record, i + epoch * total_step)

                """ Loss G """
                if self.use_visual_gan:
                    gan_lambda_handler.update_gan_lambda(epoch, i, cap_loss_record)
                    # f_caption = tokens * seq_mask
                    # f_caption = self.model.decoder.output2wordembedding(tokens)
                    object_psl = object_psl.detach()
                    motion_psl = motion_psl.detach()
                    alpha_all = alpha_all.detach()
                    f_logit = self.D_visual(tokens, object_psl, motion_psl, att_mask=att_mask, alpha_all=alpha_all)
                    loss_G = -f_logit.mean()

                    loss_G_record = loss_G.item()
                    if self.multi_gpu:
                        loss_G_record = self.reduce_tensor(loss_G.data).item()
                    loss_count_G += loss_G_record

                    gan_lambda = gan_lambda_handler.get_current_lambda()
                    # gan_lambda = 0.001
                    if self.local_rank <= 0:
                        self.writer.add_scalar('Loss/G_v_loss', loss_G_record, i + epoch * total_step)
                        self.writer.add_scalar('parameter/gan_lambda', gan_lambda, i + epoch * total_step)
                    total_loss = total_loss + loss_G * gan_lambda

                total_loss.backward()
                optimizer.step()
                """ Evaluation """
                if i % 10 == 0:
                # if i % 10 == 0 or bsz < self.train_batch_size:
                    loss_count /= 10. if bsz == self.train_batch_size else i % 10
                    results_string = f'Epoch [{epoch}/{self.epoch_num}], Step [{i}/{total_step}], Loss: {loss_count:.4f}, Perplexity: {np.exp(loss_count):.4f}'
                    loss_count = 0
                    if self.use_visual_gan:
                        loss_count_G /= 10. if bsz == self.train_batch_size else i % 10
                        loss_count_D /= 10. if bsz == self.train_batch_size else i % 10
                        results_string += f', loss_G: {loss_count_G:.4f}, loss_D: {loss_count_D:.4f}'
                        loss_count_G = 0
                        loss_count_D = 0
                    if self.local_rank <= 0:
                        print(results_string)

                    tokens = tokens.max(2)[1]
                    tokens = tokens.data[0].squeeze()

                    if self.multi_gpu:
                        we = self.model.module.decoder.decode_tokens(tokens)
                        gt = self.model.module.decoder.decode_tokens(captions[0].squeeze())
                    else:
                        we = self.model.decoder.decode_tokens(tokens)
                        gt = self.model.decoder.decode_tokens(captions[0].squeeze())
                    if self.local_rank <= 0:
                        print('[vid:%d]' % video_ids[0])
                        print('WE: %s\nGT: %s' % (we, gt))
                if i in saving_schedule:
                    start_time_eval = time.time()
                    self.model.eval()
                    metrics_list = []
                    results_list = []
                    alpha_list = []
                    if self.multi_gpu:
                        results, alpha_all = gather_results(self.model, self.args, self.test_loader, multi_modal=True, multi_gpu=self.multi_gpu)
                        results_multi = [None for _ in range(4)]
                        distributed.all_gather_object(results_multi, results)

                        if self.local_rank == 0:
                            num_keys = [len(results.keys()) for results in results_multi]
                            print('num_keys_sum = ', sum(num_keys))
                            results_all = {**results_multi[0],**results_multi[1],**results_multi[2],**results_multi[3]}
                            print('num_keys_merge = ', len(results_all.keys()))
                            blockPrint()
                            # results_all = collections.OrderedDict(sorted(results_all.items()))
                            metrics, results, i_time = evaluate_multi_gpu(results_all, self.test_reference)
                            enablePrint()
                    else:
                        blockPrint()
                        metrics, results, alpha_all, i_time = evaluate(self.model, self.args, self.test_loader, self.test_reference, multi_modal=True, multi_gpu=self.multi_gpu)
                        enablePrint()

                    if self.local_rank <= 0:
                        print(self.local_rank)
                        print('hehe -= ', len(results.keys()))
                        metrics_list.append(metrics)
                        results_list.append(results)
                        alpha_list.append(alpha_all)
                        end_time_eval = time.time()
                        print('evaluate time: %.3fs' % (end_time_eval - start_time_eval))
                        print('evaluate inference time: %.3fs' % i_time)
                        self.writer.add_scalar('results/Bleu_4', metrics_list[0]['Bleu_4'], i + epoch * total_step)
                        self.writer.add_scalar('results/METEOR', metrics_list[0]['METEOR'], i + epoch * total_step)
                        self.writer.add_scalar('results/CIDEr', metrics_list[0]['CIDEr'], i + epoch * total_step)
                        self.writer.add_scalar('results/ROUGE_L', metrics_list[0]['ROUGE_L'], i + epoch * total_step)
                        self.result_handler.update_result(metrics_list, results_list, epoch)
                        # save checkpoint
                        if self.multi_gpu:
                            torch.save({
                                'epoch': epoch,
                                'model_state_dict': self.model.module.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'model_d_state_dict': self.D_visual.module.state_dict(),
                                'optimizer_d_state_dict': optimizer_D.state_dict(),
                                'cap_list': np.array(gan_lambda_handler.cap_list),
                            }, f'{epoch}.pt')

                    self.model.train()

            end_time = time.time()
            scheduler.step()
            if self.use_visual_gan:
                scheduler_D.step()
            self.result_handler.print_results()
            print("*******One epoch time: %.3fs*******\n" % (end_time - start_time))
        self.result_handler.end_round()

    def psl_distance(self, psl, criterion_ppl):
        bs, num_ppl, h_size = psl.size()

        repeated_in_chunks = psl.repeat_interleave(num_ppl, dim=1)
        repeated_alternating = psl.repeat(1, num_ppl, 1)
        idx = []
        for i in range(1, self.num_psl):
            idx += [j + (i - 1) * self.num_psl for j in range(i, self.num_psl)]
        repeated_in_chunks = repeated_in_chunks[:, idx, :]
        repeated_alternating = repeated_alternating[:, idx, :]

        loss_ppl = criterion_ppl(repeated_in_chunks.view(-1, h_size),
                                 repeated_alternating.view(-1, h_size),
                                 torch.Tensor(bs * len(idx)).to(self.device).fill_(-1.0))
        return loss_ppl * 0.5


    def train_disc(self, r_caption, f_caption, optimizer_D, D, num_D, i, epoch, total_step,
                   loss_count_D, wasserstein, att_mask=None, obj_psl=None, motion_psl=None, pos_tag=None, alpha_all=None):
        mean_iteration_D_loss = 0
        mean_wasserstein = 0
        for _ in range(num_D):
            optimizer_D.zero_grad()
            # discriminator output
            # with torch.cuda.amp.autocast():
            if obj_psl is None:
                r_logit = D(r_caption, att_mask)
                f_logit = D(f_caption, att_mask)
            else:
                r_logit = D(r_caption, obj_psl, motion_psl, att_mask, alpha_all)
                f_logit = D(f_caption, obj_psl, motion_psl, att_mask, alpha_all)

            # calculate the gradient for penalty
            epsilon_gp = torch.rand(len(r_logit), 1, 1, device=self.device, requires_grad=True)
            mixed_captions = r_caption.detach() * epsilon_gp + f_caption.detach() * (1 - epsilon_gp)
            if obj_psl is None:
                mixed_logit = D(mixed_captions, att_mask)
            else:
                mixed_logit = D(mixed_captions, obj_psl, motion_psl, att_mask, alpha_all)

            gradient_for_gp = torch.autograd.grad(
                inputs=mixed_captions,
                outputs=mixed_logit,
                grad_outputs=torch.ones_like(mixed_logit),
                create_graph=True,
                retain_graph=True
            )[0]
            gradient_for_gp = gradient_for_gp.contiguous().view(len(gradient_for_gp), -1)
            gradient_norm = gradient_for_gp.norm(2, dim=1)
            gradient_penalty = ((gradient_norm - 1) * (gradient_norm - 1)).mean()
            # discriminator loss
            r_loss = r_logit.mean()
            f_loss = f_logit.mean()
            loss_D = f_loss - r_loss + 10 * gradient_penalty
            # update model
            mean_iteration_D_loss += loss_D.data / num_D
            mean_wasserstein += (r_loss.data - f_loss.data) / num_D
            # scaler.scale(loss_D).backward()
            loss_D.backward(retain_graph=True)
            # scaler.step(optimizer_D)
            # scaler.update()
            optimizer_D.step()

        mean_iteration_D_loss_record = mean_iteration_D_loss.item()
        mean_wasserstein_record = mean_wasserstein.item()
        if self.multi_gpu:
            mean_iteration_D_loss_record = self.reduce_tensor(mean_iteration_D_loss.data).item()
            mean_wasserstein_record = self.reduce_tensor(mean_wasserstein.data).item()
        loss_count_D += mean_iteration_D_loss_record
        wasserstein += mean_wasserstein_record
        loss_name = 'lang'
        if pos_tag is not None:
            loss_name = 'visual'
        if self.local_rank <= 0:
            self.writer.add_scalar(f'Loss/D_loss_{loss_name}', mean_iteration_D_loss_record, i + epoch * total_step)
            self.writer.add_scalar(f'Loss/wasserstein_{loss_name}', mean_wasserstein_record, i + epoch * total_step)
        return loss_count_D, wasserstein

    def get_visual_gan_lambda_schedule(self):
        steps = self.epoch_num
        num_repeat = 3

        steps /= 3
        schedule = torch.linspace(0.0001, 0.001, steps=math.ceil(steps)).to(self.device)
        schedule = schedule.repeat_interleave(num_repeat)

        return schedule

    def train_generator(self, optim_G, batch_size):
        pass

    def get_trainer_name(self, args):
        self.ss_factor = args.ss_factor
        self.num_psl = args.num_proposals
        base_name = f'{args.dataset}_{args.ss_factor}_GNN'
        base_name += f'_{args.num_obj}_{args.num_proposals}'
        print(base_name)
        if args.use_psl_loss:
            base_name += '_use_psl_loss'

        self.use_visual_gan = args.use_visual_gan
        if args.use_visual_gan:
            self.num_D_visual = args.num_D_visual
            self.lambda_D_visual = args.lambda_D_visual
            visual_name = f'_visual_{self.lambda_D_visual}_{self.num_D_visual}'
            if self.local_rank <= 0:
                print(visual_name)
            base_name += visual_name

        return base_name

    def reduce_tensor(self, tensor):
        rt = tensor.clone()
        distributed.all_reduce(rt, op=distributed.ReduceOp.SUM)
        rt /= distributed.get_world_size()
        return rt

    def test(self):
        reference = convert_data_to_coco_scorer_format(self.args.test_reference_txt_path)
        metrics = evaluate(self.model, self.args, self.test_loader, reference)

    def make_one_hot_encoding(self, seq, vocab_size):
        sent_onehot = torch.zeros(seq.size(0), vocab_size).to(self.device)
        sent_onehot.scatter_(1,seq,1)
        sent_onehot[:,0] = 0
        return sent_onehot

    def to_onehot(self, seq, vocab_size):
        batch_size, seq_len = seq.size(0), seq.size(1)
        onehot = torch.zeros(batch_size, seq_len, vocab_size).to(self.device)
        onehot.scatter_(2, seq.unsqueeze(2), 1)
        return onehot

    def plot_alpha_all(self, alpha_all, title, epoch, i, vid):
        alpha_obj = alpha_all[0, :, :self.num_psl].detach().cpu().numpy()
        alpha_mt = alpha_all[0, :, -self.num_psl:].detach().cpu().numpy()
        fig, ax = plt.subplots(1, 2)
        alpha_obj = alpha_obj/alpha_obj.max(axis=1)[:, np.newaxis]
        alpha_mt = alpha_mt/alpha_obj.max(axis=1)[:, np.newaxis]
        sns.heatmap(alpha_obj, ax=ax[0], cbar=False, yticklabels=False, xticklabels=False)
        sns.heatmap(alpha_mt, ax=ax[1], cbar=False, yticklabels=False, xticklabels=False)
        plt.title(title)
        plt.savefig(f'./images/{self.base_name}/{vid}_{epoch}_{i}.png')
        plt.close()