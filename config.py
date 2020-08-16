import os

data_folder = '/home/anna/seq_classify/data/aclImdb'
train_fname = data_folder + '/imdb_train_df.csv'
test_fname = data_folder + '/imdb_test_df.csv'


MAX_SEQ_LEN = 512
NUM_EPOCHS = 10
BATCH_SIZE = 200
LR = 3e-5
NUM_CPU_WORKERS = 20
PRINT_EVERY = 100