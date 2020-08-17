import os
import psutil

data_folder = '/home/anna/seq_classify/data/aclImdb'
train_fname = data_folder + '/imdb_train_df.csv'
test_fname = data_folder + '/imdb_test_df.csv'

MAX_SEQ_LEN = 512
NUM_EPOCHS = 20
BATCH_SIZE = 20
LR = 0.005
NUM_CPU_WORKERS = psutil.cpu_count()
PRINT_EVERY = 100
BERT_LAYER_FREEZE = False