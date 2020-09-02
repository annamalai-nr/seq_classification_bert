import os, torch, sys
import psutil

os.environ["TOKENIZERS_PARALLELISM"] = "true"
MODEL_NAME =  'bert-base-uncased'
MODEL_NAME =  'distilbert-base-uncased'

data_folder = '/home/anna/seq_classify/data/aclImdb'
train_fname = data_folder + '/imdb_train_df.csv'
test_fname = data_folder + '/imdb_test_df.csv'

MAX_SEQ_LEN = 256
NUM_EPOCHS = 20
BATCH_SIZE = 50
LR = 0.001
NUM_CPU_WORKERS = psutil.cpu_count()
PRINT_EVERY = 100
BERT_LAYER_FREEZE = False

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")
CONTEXT_VECTOR_SIZE = 1024 if 'large' in MODEL_NAME else 768
IS_LOWER = True if 'uncased' in MODEL_NAME else False

MULTIGPU = True if torch.cuda.device_count() > 1 else False #when using xlarge vs 16x large AWS m/c

