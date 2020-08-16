import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer
import pandas as pd
import numpy as np
from tqdm import tqdm

import config


class dataset(Dataset):
    def __init__(self, dataset_fname, max_len, sample_ratio=None):
        # Initialize the BERT tokenizer
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.max_len = max_len

        train_df = pd.read_csv(dataset_fname)
        if sample_ratio:
            train_df = train_df.sample(frac=sample_ratio)
        print(f'Loaded dataframe of shape: {train_df.shape}')
        self.sent_token_ids_attn_masks = [self._get_token_ids_attn_mask(s) for s in tqdm(train_df.sentence)]
        self.labels = np.array(train_df.label, dtype=int)

        print(f'Loaded X_train and y_train from {config.train_fname}, '
              f'shapes: {len(self.sent_token_ids_attn_masks), self.labels.shape}')


    def _get_token_ids_attn_mask(self, sentence):
        sentence = sentence.lower().strip()
        tokens = self.tokenizer.tokenize(sentence)  # Tokenize the sentence
        tokens = ['[CLS]'] + tokens + ['[SEP]']  # Insering the CLS and SEP token in the beginning and end of the sentence
        if len(tokens) < self.max_len:
            tokens = tokens + ['[PAD]' for _ in range(self.max_len - len(tokens))]  # Padding sentences
        else:
            tokens = tokens[:self.max_len - 1] + ['[SEP]']  # Prunning the list to be of specified max length

        tokens_ids = self.tokenizer.convert_tokens_to_ids(
            tokens)  # Obtaining the indices of the tokens in the BERT Vocabulary
        tokens_ids_tensor = torch.tensor(tokens_ids)  # Converting the list to a pytorch tensor

        # Obtaining the attention mask i.e a tensor containing 1s for no padded tokens and 0s for padded ones
        attn_mask = (tokens_ids_tensor != 0).long()

        return tokens_ids_tensor, attn_mask

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        #Selecting the sentence and label at the specified index in the data frame
        token_ids,attn_mask = self.sent_token_ids_attn_masks[index] #list index
        label = self.labels[index] #array index

        return token_ids, attn_mask, label
