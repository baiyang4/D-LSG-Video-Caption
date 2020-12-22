import os, sys
import torch

class Vocabulary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []
        self.nwords = 0
        self.add_word('<pad>')
        self.add_word('<start>')
        self.add_word('<end>')
        self.add_word('<unk>')

    def add_word(self, w):
        '''
        add a new word to the vocabulary
        '''
        if w not in self.word2idx:
            self.word2idx[w] = self.nwords
            self.idx2word.append(w)
            self.nwords += 1

    def __call__(self, w):
        '''
        :return corresponding index of the given word
        '''
        if w not in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[w]

    def __len__(self):
        '''
        get number of words in the vocabulary
        '''
        return self.nwords


def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)

def blockPrint():
    sys.stdout = open(os.devnull, 'w')

def enablePrint():
    sys.stdout = sys.__stdout__

class ResultHandler():
    def __init__(self, model, args, is_debug=True):
        super(ResultHandler, self).__init__()
        self.model_saving = model
        self.path = f'./models_saved/{args.dataset}/{args.dropout}_{args.use_glove}'
        self.path_temp = '{}/temp'.format(self.path)
        self.on = (is_debug is False)
        self.bleu_best = 0
        self.best_10 = []
        if self.on:
            self.start()

    def start(self):
        os.makedirs(self.path_temp, exist_ok=True)
        os.makedirs(self.path, exist_ok=True)
        self._empty_path(self.path)

    @staticmethod
    def _empty_path(path):
        for the_file in os.listdir(path):
            file_path = os.path.join(path, the_file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                # elif os.path.isdir(file_path): shutil.rmtree(file_path)
            except Exception as e:
                print(e)

    def _move_model(self):
        for the_file in os.listdir(self.path_temp):
            file_path = os.path.join(self.path_temp, the_file)
            try:
                if os.path.isfile(file_path):
                    os.rename(file_path, '{}/{}'.format(self.path, the_file))
            except Exception as e:
                print(e)

    def update_result(self, metrics):
        for k, v in metrics.items():
            # log_value(k, v, epoch * len(saving_schedule) + count)
            print('%s: %.6f' % (k, v))

            if k =='Bleu_4' and v > self.bleu_best:
                self.bleu_best = v
                # print('best result: ', self.bleu_best)
                if self.on:
                    self._empty_path(self.path_temp)
                    torch.save(self.model_saving.state_dict(), f'{self.path_temp}/{v:.4f}')
                    print(f'save model {v:.4f}')
        print('best result: ', self.bleu_best)

    def end_round(self):
        if self.on:
            self._move_model()
            os.rmdir(self.path_temp)
            print('best result: ', self.bleu_best)
