import torch
import transformers
import pandas as pd
from pprint import pprint
import numpy as np
import time
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim



import config
import dataset
import model

def get_accuracy_from_logits(logits, labels):
    soft_probs = logits.argmax(1).long()
    acc = (soft_probs.squeeze() == labels).float().mean()
    return acc


def test(net, test_loader, device='cpu'):
    net.eval()
    with torch.no_grad():
        accs = []
        for it, (seq, attn_masks, labels) in enumerate(test_loader):
            seq, attn_masks, labels = seq.cuda(device), attn_masks.cuda(device), labels.cuda(device)

            logits = net(seq, attn_masks)
            preds = logits.argmax(1)
            acc = (preds.squeeze() == labels).float().mean()
            accs.append(acc.item())
    net.train()
    accs = np.array(accs).mean()
    return accs


def train_model(net, criterion, opti, train_loader, test_loader=None, print_every=100, n_epochs=10, device='cpu'):
    for e in range(1, n_epochs+1):
        t0 = time.perf_counter()
        e_loss = []
        for batch_num, (seq_attnmask_labels) in enumerate(train_loader):
            # Clear gradients
            opti.zero_grad()

            seq, attn_mask, labels = seq_attnmask_labels
            # Converting these to cuda tensors
            seq, attn_mask, labels = seq.cuda(device), attn_mask.cuda(device), labels.cuda(device)

            # Obtaining the logits from the model
            logits = net(seq, attn_mask)

            # Computing loss
            loss = criterion(logits.squeeze(-1), labels.long()) #need to change credit_labels
            e_loss.append(loss.item())
            # Backpropagating the gradients
            loss.backward()

            # Optimization step
            opti.step()

            if (batch_num + 1) % print_every == 0:
                acc = get_accuracy_from_logits(logits, labels)
                print(f"batch {batch_num+1} of epoch {e} complete. Loss : {loss.item()} Accuracy : {acc}")

        t = time.perf_counter() - t0
        e_loss = np.array(e_loss).mean()
        print(f'Done epoch: {e} in {round(t,2)} sec, epoch loss: {e_loss}')
        if test_loader != None:
            test_acc = test(net, test_loader, device=device)
            print(f'After epoch: {e}, test accuracy: {test_acc}')


def main():
    device = f'cuda:0'
    # Creating instances of training and validation set
    train_set = dataset.dataset(dataset_fname = '/home/anna/seq_classify/data/aclImdb/imdb_train_df.csv', max_len=config.MAX_SEQ_LEN)
    test_set = dataset.dataset(dataset_fname = '/home/anna/seq_classify/data/aclImdb/imdb_test_df.csv', max_len=config.MAX_SEQ_LEN)

    train_loader = DataLoader(train_set, shuffle = True,
                              batch_size=config.BATCH_SIZE, num_workers=config.NUM_CPU_WORKERS)
    test_loader = DataLoader(test_set, shuffle = True,
                             batch_size=config.BATCH_SIZE, num_workers=config.NUM_CPU_WORKERS)

    bert_model = model.bert_classifier(freeze_bert=True)
    bert_model.to(device)

    print(f"created BERT model for finetuning: {bert_model}")
    criterion = nn.CrossEntropyLoss()
    opti = optim.Adam(bert_model.parameters(), lr=config.LR)

    train_model(bert_model, criterion, opti, train_loader, test_loader=test_loader, print_every=config.PRINT_EVERY,
                n_epochs=config.NUM_EPOCHS, device=device)


if __name__ == '__main__':
    main()