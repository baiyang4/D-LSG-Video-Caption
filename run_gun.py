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
from models.model import CapModel, Disc


class RunGAN:
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
        # create model
        self.model = CapModel(args, vocab).to(device)
        if model_path is not None:
            self.model.load_state_dict(torch.load(model_path, map_location='cuda:0'))
        # parameters for training
        self.D = Disc(args).to(device)
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
        criterion_D = nn.BCELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        optimizer_D = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        lr_steps = [1,2,3,18]
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_steps, gamma=0.5)
        scheduler_D = torch.optim.lr_scheduler.MultiStepLR(optimizer_D, milestones=lr_steps, gamma=0.3)

        total_step = len(self.train_loader)

        loss_count = 0
        loss_count_G = 0
        loss_count_D = 0
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

                """ Train D """
                mean_iteration_D_loss = 0
                for _ in range(5):
                    optimizer_D.zero_grad()
                    r_caption = self.model.decoder.caption2wordembedding(targets)
                    f_caption = self.model(frames, targets, max_len, epsilon)
                    f_caption = self.model.decoder.output2wordembedding(f_caption.detach()).detach()

                    # discriminator output
                    r_logit = self.D(r_caption)
                    f_logit = self.D(f_caption)

                    # calculate the gradient for penalty
                    epsilon_gp = torch.rand(len(r_logit), 1, 1, device=self.device, requires_grad=True)
                    mixed_captions = r_caption.detach() * epsilon_gp + f_caption.detach() * (1 - epsilon_gp)
                    mixed_logit = self.D(mixed_captions)
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
                    mean_iteration_D_loss += loss_D.item() / 5
                    loss_D.backward()
                    optimizer_D.step()
                loss_count_D += mean_iteration_D_loss
                """ Train G """

                optimizer.zero_grad()
                outputs = self.model(frames, targets, max_len, epsilon)
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

                """ Loss G """
                f_caption = self.model.decoder.output2wordembedding(tokens)
                f_logit = self.D(f_caption)
                loss_G = -f_logit.mean()

                # add loss
                loss_count += cap_loss.item()
                loss_count_G += loss_G.item()

                total_loss = cap_loss + loss_G * 0.2
                total_loss.backward()
                # clip_gradient(optimizer, self.args.grad_clip)
                optimizer.step()

                if i % 10 == 0 or bsz < self.train_batch_size:
                    loss_count /= 10 if bsz == self.train_batch_size else i % 10
                    print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f, loss_G: %.4f, loss_D: %.4f, Perplexity: %5.4f' %
                          (epoch, self.epoch_num, i, total_step, loss_count, loss_count_G, loss_count_D,
                           np.exp(loss_count)))

                    loss_count = 0
                    loss_count_G = 0
                    loss_count_D = 0
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
                    metrics = evaluate(self.model, self.args, self.test_loader, self.test_reference)
                    end_time_eval = time.time()
                    enablePrint()
                    print('evaluate time: %.3fs' % (end_time_eval - start_time_eval))

                    self.result_handler.update_result(metrics)
                    self.model.train()

            end_time = time.time()
            scheduler.step()
            scheduler_D.step()
            print("*******One epoch time: %.3fs*******\n" % (end_time - start_time))
        self.result_handler.end_round()

    def train_generator(self, optim_G, batch_size):
        pass

    def test(self):
        reference = convert_data_to_coco_scorer_format(self.args.test_reference_txt_path)
        metrics = evaluate(self.model, self.args, self.test_loader, reference)
