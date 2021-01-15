import os, sys
import torch
import pandas as pd
import getpass

SAVING_MODEL_NAME = ''

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
    def __init__(self, model, base_name, is_debug=True):
        super(ResultHandler, self).__init__()
        self.model_saving = model
        if type(base_name) is not str:
            base_name = 'GNN_base'
        self.path = f'./models_saved/{base_name}/{getpass.getuser()}'
        self.path_results = f'./results/{base_name}/{getpass.getuser()}'

        # self.path_temp = '{}/temp'.format(self.path)
        self.on = (is_debug is False)
        self.results_recorder = ResultsRecorder([5], self.path_results)
        # if self.on:
        self.start()

    def start(self):
        # os.makedirs(self.path_temp, exist_ok=True)
        os.makedirs(self.path, exist_ok=True)
        # self._empty_path(self.path)

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

    def update_result(self, metrics, results, epoch=0):
        if type(metrics) is not list:
            metrics = [metrics]
            results = [results]
        should_save = self.results_recorder.update_results(metrics, results, epoch)
        if should_save:
            self.results_recorder.save_results(self.path_results)
        global SAVING_MODEL_NAME
        if len(SAVING_MODEL_NAME) > 0 and self.on:
            torch.save(self.model_saving.state_dict(), f'{self.path}/{SAVING_MODEL_NAME}')
            print(f'save model {SAVING_MODEL_NAME}')
            SAVING_MODEL_NAME = ''


    def print_results(self):
        self.results_recorder.print_results()

    def end_round(self):
        if self.on:
            self._move_model()
            os.rmdir(self.path_temp)
            print('best result: ', self.bleu_best)


class DataRecorder:
    def __init__(self, beam_size, path):
        self.beam_size = beam_size
        self.path = path
        self.record_dic = {'Bleu_4': 0, 'METEOR': 0, 'CIDEr': 0, 'ROUGE_L': 0}
        self.record_epoch_dic = {'Bleu_4': 0, 'METEOR': 0, 'CIDEr': 0, 'ROUGE_L': 0}

    def update_results(self, metrics, results, epoch):
        should_save = False
        for k, v in metrics.items():
            # if self.beam_size == 3:
            print('%s: %.6f' % (k, v))
            if k in self.record_dic:
                current_v = self.record_dic[k]
                if v > current_v:
                    should_save = True
                    if k == 'Bleu_4' or k == 'CIDEr':
                        global SAVING_MODEL_NAME
                        SAVING_MODEL_NAME = k
                    self.record_dic[k] = v
                    self.record_epoch_dic[k] = epoch
                    results_df = pd.DataFrame(results, index=[0]).T.reset_index()
                    results_df.columns = ['vid', 'pred']
                    results_df[['vid']] = results_df[['vid']].astype(int)
                    results_df.to_csv(f'{self.path}/{k}_{self.beam_size}.csv', index=False)
        # self.print_results()
        return should_save

    def print_results(self):
        print('--------------beam_size = ', self.beam_size)
        for key in self.record_dic.keys():
            print(f'{key}:{self.record_dic[key]:.3f}, epoch {self.record_epoch_dic[key]}')
        print('--------------')


class ResultsRecorder:
    def __init__(self, beam_list, path):
        self.beam_list = beam_list
        path_captioning = f'{path}/captioning'
        os.makedirs(path, exist_ok=True)
        os.makedirs(path_captioning, exist_ok=True)
        self.data_recorders = [DataRecorder(beam_size, path_captioning) for beam_size in beam_list]

    def update_results(self, metrics_list, results_list, epoch):
        should_save = False
        for i in range(len(self.beam_list)):
            data_recorder = self.data_recorders[i]
            metrics = metrics_list[i]
            results = results_list[i]
            should_save = data_recorder.update_results(metrics, results, epoch)
        return should_save

    def save_results(self, save_path):
        dict_list = [recorder.record_dic for recorder in self.data_recorders]
        df = pd.DataFrame(dict_list)
        df = df.round(4)
        df.to_csv(f'{save_path}/metrics.csv')
        print('results saved')

    def print_results(self):
        for data_recorder in self.data_recorders:
            data_recorder.print_results()

