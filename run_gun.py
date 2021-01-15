
import time
from utils.utils import *

import torch
import torch.nn as nn
import numpy as np
from evaluate import evaluate, convert_data_to_coco_scorer_format
from models.model import Disc, CapGnnModel, DiscV
from torch.utils.tensorboard import SummaryWriter
import math


class RunGAN:
    def __init__(self, args, vocab, device, train_loader=None, test_loader=None, test_reference=None, is_debug=True, model_path=None):
        # load training data
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.test_reference = test_reference
        print('dropout = ', args.dropout)
        self.use_graph = args.use_graph

        self.base_name = self.get_trainer_name(args)

        self.num_D_switch = args.num_D_switch
        print('num_D_switch = ', args.num_D_switch)

        vocab_size = len(vocab)
        print(vocab_size)
        print('use_graph = ', args.use_graph)

        # create model

        self.model = CapGnnModel(args, vocab).to(device)

        if model_path is not None:
            # model = CapGnnModel(args, vocab).to(device)
            # model.load_state_dict(torch.load(model_path, map_location='cuda:0'))
            self.model.load_encoder(CapGnnModel(args, vocab).to(device), model_path)
        # parameters for training
        if self.use_lang_gan:
            # self.D_lang = DiscLanguage(args, vocab_size).to(device)
            self.D_lang = Disc(args).to(device)
        if self.use_visual_gan:
            self.D_visual = DiscV(args, vocab_size).to(device)

        self.args = args
        self.lr = args.learning_rate
        self.epoch_num = args.epoch_num
        self.device = device
        self.train_batch_size = args.train_batch_size
        self.test_batch_size = args.test_batch_size
        self.save_per_epoch = args.save_per_epoch
        self.vocab_size = vocab_size
        self.ss_factor = args.ss_factor
        self.result_handler = ResultHandler(self.model, self.base_name, is_debug=is_debug)
        self.writer = SummaryWriter(comment=self.base_name)

    def train(self):
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, betas=(0.5, 0.9))
        # optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        # lr_steps = [1,2,3,18]
        # lr_steps = [1, 4]
        lr_steps = [1, 4]
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_steps, gamma=0.5)

        if self.use_lang_gan:
            lr_steps_D = [1, 4]
            optimizer_D = torch.optim.Adam(self.D_lang.parameters(), lr=self.lr, betas=(0.5, 0.9))
            scheduler_D = torch.optim.lr_scheduler.MultiStepLR(optimizer_D, milestones=lr_steps_D, gamma=0.3)

        if self.use_visual_gan:
            lr_steps_D_v = [1, 4]
            optimizer_D_v = torch.optim.Adam(self.D_visual.parameters(), lr=self.lr, betas=(0.5, 0.9))
            scheduler_D_v = torch.optim.lr_scheduler.MultiStepLR(optimizer_D_v, milestones=lr_steps_D_v, gamma=0.5)
        total_step = len(self.train_loader)

        loss_count = 0
        loss_count_G = 0
        loss_count_D = 0
        wasserstein = 0
        loss_count_G_v = 0
        loss_count_D_v = 0
        wasserstein_v = 0

        saving_schedule_small = [int(x * total_step / 2) for x in list(range(1, 2 + 1))]
        saving_schedule_mid = [int(x * total_step / 8) for x in list(range(1, 8 + 1))]
        saving_schedule_high = [int(x * total_step / 16) for x in list(range(1, 16 + 1))]
        # saving_schedule = [20,200,300]
        print('total: ', total_step)
        # print('saving_schedule: ', saving_schedule)

        gan_switch_training = self.use_lang_gan and self.use_visual_gan and self.num_D_switch > 0
        if self.use_visual_gan:
            visual_gan_lambda = self.get_visual_gan_lambda_schedule()
            print(len(visual_gan_lambda))
            print(visual_gan_lambda)
        for epoch in range(self.epoch_num):
            # if epoch > 4:
            #     saving_schedule = [int(x * total_step / 12) for x in
            #                        list(range(1, 12 + 1))]
            # if epoch > 8:
            #     saving_schedule = [int(x * total_step / 16) for x in
            #                        list(range(1, 16 + 1))]
            start_time = time.time()
            epsilon = max(0.6, self.ss_factor / (self.ss_factor + np.exp(epoch / self.ss_factor)))
            print('Epoch-{0} lr: {1}'.format(epoch, optimizer.param_groups[0]['lr']))
            if self.use_lang_gan:
                print('Epoch-{0} lr language GAN: {1}'.format(epoch, optimizer_D.param_groups[0]['lr']))
            if self.use_visual_gan:
                print('Epoch-{0} lr visual GAN: {1}'.format(epoch, optimizer_D_v.param_groups[0]['lr']))
                print('visual gan lambda: ', visual_gan_lambda[epoch])
            for i, (frames, regions, spatials, captions, pos_tags, cap_lens, video_ids) in enumerate(self.train_loader, start=1):

                max_len = 26
                bs = frames.shape[0]

                frames = frames.to(self.device)
                # batch_size * sentence_len
                regions = regions[:, :, :self.args.num_obj, :].to(self.device)
                captions = captions[:, :max_len]
                targets = captions.to(self.device)
                if self.use_visual_gan:
                    pos_tags = pos_tags.to(self.device)

                # visual gan, language gan, each train 5 times individually
                if gan_switch_training:
                    if i % self.num_D_switch * 2 < self.num_D_switch:
                        self.use_lang_gan = True
                        self.use_visual_gan = False
                    else:
                        self.use_lang_gan = False
                        self.use_visual_gan = True
                """ Train D """
                if self.use_lang_gan or self.use_visual_gan:

                    seq_mask = (captions > 0).to(torch.float32)
                    att_mask = torch.matmul(seq_mask.view(bs, max_len, 1), seq_mask.view(bs, 1, max_len)).to(
                        self.device)
                    f_caption, object_psl, motion_psl, alpha_all = self.model(frames, regions, targets, max_len, epsilon)

                if self.use_lang_gan:

                    r_caption = self.model.decoder.caption2wordembedding(targets).detach()
                    f_caption = self.model.decoder.output2wordembedding(f_caption.detach()).detach()
                    # f_caption = f_caption.detach()
                    # r_caption = self.to_onehot(targets, self.vocab_size)
                    loss_count_D, wasserstein = \
                        self.train_disc(r_caption, f_caption, optimizer_D, self.D_lang, self.num_D_lang, i, epoch,
                                        total_step, loss_count_D, wasserstein, att_mask)

                if self.use_visual_gan:
                    f_caption_v = f_caption.detach()
                    r_caption_v = self.to_onehot(targets, self.vocab_size)
                    object_psl = object_psl.detach()
                    motion_psl = motion_psl.detach()
                    alpha_all = alpha_all.detach()
                    loss_count_D_v, wasserstein_v = \
                        self.train_disc(r_caption_v, f_caption_v, optimizer_D_v, self.D_visual, self.num_D_visual,
                                        i, epoch, total_step, loss_count_D_v, wasserstein_v, obj_psl=object_psl,
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

                loss_count += cap_loss.item()
                self.writer.add_scalar('Loss/cap_loss', cap_loss.item(), i + epoch * total_step)

                """ Loss G """

                if self.use_lang_gan:
                    f_caption = self.model.decoder.output2wordembedding(tokens)
                    f_logit = self.D_lang(f_caption, att_mask)
                    loss_G = -f_logit.mean()
                    loss_count_G += loss_G.item()
                    self.writer.add_scalar('Loss/G_loss', loss_G.item(), i + epoch * total_step)
                    total_loss = total_loss + loss_G * self.lambda_D_lang

                if self.use_visual_gan:
                    # f_caption = tokens * seq_mask
                    # f_caption = self.model.decoder.output2wordembedding(tokens)
                    object_psl = object_psl.detach()
                    motion_psl = motion_psl.detach()
                    alpha_all = alpha_all.detach()
                    f_logit = self.D_visual(tokens, pos_tags, object_psl, motion_psl, att_mask=att_mask, alpha_all=alpha_all)
                    loss_G_v = -f_logit.mean()
                    loss_count_G_v += loss_G_v.item()
                    self.writer.add_scalar('Loss/G_v_loss', loss_G_v.item(), i + epoch * total_step)
                    # total_loss = total_loss + loss_G_v * 0.0006
                    total_loss = total_loss + loss_G_v * self.lambda_D_visual

                total_loss.backward()
                # scaler.scale(total_loss).backward()
                # scaler.step(optimizer)
                optimizer.step()
                # scaler.update()
                if i % 10 == 0 or bsz < self.train_batch_size:
                    loss_count /= 10. if bsz == self.train_batch_size else i % 10
                    results_string = f'Epoch [{epoch}/{self.epoch_num}], Step [{i}/{total_step}], Loss: {loss_count:.4f}, Perplexity: {np.exp(loss_count):.4f}'

                    loss_count = 0
                    if self.use_lang_gan:
                        results_string += f', loss_G: {loss_count_G:.4f}, loss_D: {loss_count_D:.4f}'
                        loss_count_G = 0
                        loss_count_D = 0
                    if self.use_visual_gan:
                        results_string += f', loss_G_v: {loss_count_G_v:.4f}, loss_D_v: {loss_count_D_v:.4f}'
                        loss_count_G_v = 0
                        loss_count_D_v = 0

                    print(results_string)

                    tokens = tokens.max(2)[1]
                    tokens = tokens.data[0].squeeze()

                    we = self.model.decoder.decode_tokens(tokens)
                    gt = self.model.decoder.decode_tokens(captions[0].squeeze())

                    print('[vid:%d]' % video_ids[0])
                    print('WE: %s\nGT: %s' % (we, gt))

                if epoch < 4:
                    saving_schedule = saving_schedule_small
                elif epoch < 8:
                    saving_schedule = saving_schedule_high
                else:
                    saving_schedule = saving_schedule_high

                if i in saving_schedule:

                    blockPrint()
                    start_time_eval = time.time()
                    self.model.eval()
                    beam_list = [5]
                    metrics_list = []
                    results_list = []
                    for beam_size in beam_list:
                        self.model.update_beam_size(beam_size)
                        # with torch.cuda.amp.autocast():
                        metrics, results = evaluate(self.model, self.args, self.test_loader, self.test_reference, multi_modal=self.use_graph)
                        metrics_list.append(metrics)
                        results_list.append(results)
                    # end_time_eval = time.time()

                    end_time_eval = time.time()
                    enablePrint()
                    print('evaluate time: %.3fs' % (end_time_eval - start_time_eval))
                    self.writer.add_scalar('results/Bleu_4', metrics_list[0]['Bleu_4'], i + epoch * total_step)
                    self.writer.add_scalar('results/METEOR', metrics_list[0]['METEOR'], i + epoch * total_step)
                    self.writer.add_scalar('results/CIDEr', metrics_list[0]['CIDEr'], i + epoch * total_step)
                    self.writer.add_scalar('results/ROUGE_L', metrics_list[0]['ROUGE_L'], i + epoch * total_step)
                    self.result_handler.update_result(metrics_list, results_list, epoch)
                    self.model.train()
                    # self.model.encoder.eval()

            end_time = time.time()
            scheduler.step()
            if self.use_lang_gan:
                scheduler_D.step()
            if self.use_visual_gan:
                scheduler_D_v.step()
            self.result_handler.print_results()
            print("*******One epoch time: %.3fs*******\n" % (end_time - start_time))
        self.result_handler.end_round()

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
                r_logit = D(r_caption, pos_tag, obj_psl, motion_psl, att_mask, alpha_all)
                f_logit = D(f_caption, pos_tag, obj_psl, motion_psl, att_mask, alpha_all)

            # calculate the gradient for penalty
            epsilon_gp = torch.rand(len(r_logit), 1, 1, device=self.device, requires_grad=True)
            mixed_captions = r_caption.detach() * epsilon_gp + f_caption.detach() * (1 - epsilon_gp)
            if obj_psl is None:
                mixed_logit = D(mixed_captions, att_mask)
            else:
                mixed_logit = D(mixed_captions, pos_tag, obj_psl, motion_psl, att_mask, alpha_all)

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
            mean_iteration_D_loss += loss_D.item() / num_D
            mean_wasserstein += (r_loss.item() - f_loss.item()) / num_D
            # scaler.scale(loss_D).backward()
            loss_D.backward(retain_graph=True)
            # scaler.step(optimizer_D)
            # scaler.update()
            optimizer_D.step()
        loss_count_D += mean_iteration_D_loss
        wasserstein += mean_wasserstein
        loss_name = 'lang'
        if pos_tag is not None:
            loss_name = 'visual'
        self.writer.add_scalar(f'Loss/D_loss_{loss_name}', mean_iteration_D_loss, i + epoch * total_step)
        self.writer.add_scalar(f'Loss/wasserstein_{loss_name}', mean_wasserstein, i + epoch * total_step)
        return loss_count_D, wasserstein

    def get_visual_gan_lambda_schedule(self):
        steps = self.epoch_num
        num_repeat = 3
        if self.use_visual_gan and self.use_lang_gan and self.num_D_switch > 0:
            steps /= self.num_D_switch
            num_repeat = self.num_D_switch
        else:
            steps /= 3
        schedule = torch.linspace(0.0005, 0.003, steps=math.ceil(steps)).to(self.device)
        schedule = schedule.repeat_interleave(num_repeat)

        return schedule

    def train_generator(self, optim_G, batch_size):
        pass

    def get_trainer_name(self, args):
        self.ss_factor = args.ss_factor
        base_name = f'{args.dataset}_{args.ss_factor}_GNN'
        base_name += f'_{args.num_obj}_{args.num_proposals}'
        print(base_name)

        self.use_lang_gan = args.use_lang_gan
        if args.use_lang_gan:
            self.num_D_lang = args.num_D_lang
            self.lambda_D_lang = args.lambda_D_lang
            lang_name = f'_lang_{self.lambda_D_lang}_{self.num_D_lang}'
            print(lang_name)
            base_name += lang_name

        self.use_visual_gan = args.use_visual_gan
        if args.use_visual_gan:
            self.num_D_visual = args.num_D_visual
            self.lambda_D_visual = args.lambda_D_visual
            visual_name = f'_visual_{self.lambda_D_visual}_{self.num_D_visual}'
            print(visual_name)
            base_name += visual_name

        return base_name


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