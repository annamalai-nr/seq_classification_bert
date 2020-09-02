import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import pandas as pd
import numpy as np
from tqdm import tqdm

import config


class dataset(Dataset):
    def __init__(self, dataset_fname, model_name, max_len, is_lower, sample_ratio=None):
        self.max_len = max_len
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        print(f'Set max seq. len: {self.max_len} for tokenizer: {self.tokenizer}')


        df = pd.read_csv(dataset_fname)
        if sample_ratio: df = df.sample(frac=sample_ratio)
        print(f'Loaded dataframe of shape: {df.shape}')
        self.sent_token_ids_attn_masks = [self._get_token_ids_attn_mask(s, is_lower=is_lower) for s in tqdm(df.sentence)]
        self.labels = np.array(df.label, dtype=int)

        print(f'Loaded X_train and y_train from {dataset_fname}, '
              f'shapes: {len(self.sent_token_ids_attn_masks), self.labels.shape}')


    def _get_token_ids_attn_mask(self, sentence, is_lower=False):
        sentence = str(sentence).strip()
        sentence = ' '.join(sentence.split())  # make sure unwanted spaces are removed
        if is_lower:
            sentence = sentence.lower()

        # encode_plus is better than calling tokenizer.tokenize and get the IDs later -
        # ref:Abisek Thakur youtube video
        inputs = self.tokenizer.encode_plus(sentence, None,
                                            add_special_tokens=True,
                                            max_length=self.max_len,
                                            pad_to_max_length=True,
                                            truncation=True
                                            )

        # need to convert them as tensors
        tokens_ids_tensor = torch.tensor(inputs["input_ids"], dtype=torch.long)
        attn_mask = torch.tensor(inputs["attention_mask"], dtype=torch.long)
        return tokens_ids_tensor, attn_mask


    def __len__(self):
        return len(self.labels)


    def __getitem__(self, index):
        # Selecting the sentence and label at the specified index in the data frame
        token_ids, attn_mask = self.sent_token_ids_attn_masks[index]  # list index
        label = self.labels[index]  # array index
        return token_ids, attn_mask, label
