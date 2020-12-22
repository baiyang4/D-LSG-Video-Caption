from torchtext.data import Field, BucketIterator
from torchtext.datasets import IMDB


TEXT = Field(lower=True, fix_length=270,
                 tokenize=list, batch_first=True)
LABEL = Field(sequential=False)
train_data, test_data = IMDB.splits(TEXT, LABEL)
TEXT.build_vocab(train_data)
LABEL.build_vocab(train_data)

train_iter, test_iter = BucketIterator.splits(
            (train_data, test_data), batch_size=64, repeat=True)
vocab_size = len(TEXT.vocab)


train_iter = iter(train_iter)
batch = next(train_iter)
text, label = batch.text, batch.label
print('heh')