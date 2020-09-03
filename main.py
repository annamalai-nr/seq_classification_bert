import torch
import numpy as np
import time
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score


import config
import dataset
import model

def get_accuracy_from_logits(logits, y_true):
    y_pred = logits.argmax(1).cpu().numpy()
    y_true = y_true.cpu().numpy()
    acc = accuracy_score(y_true, y_pred)
    return acc


def test(net, test_loader, device='cpu'):
    accs = []
    net.eval()
    with torch.no_grad():
        for it, (seq, attn_masks, labels) in enumerate(tqdm(test_loader), start=1):
            seq, attn_masks = seq.cuda(device), attn_masks.cuda(device)
            logits = net(seq, attn_masks)
            y_pred = logits.argmax(1).cpu().numpy()
            y_true = labels.cpu().numpy()
            a = accuracy_score(y_true, y_pred)
            accs.append(a)
    accs = np.array(accs).mean()
    net.train()
    return accs


def train_model(net, criterion, optimizer, scheduler, train_loader, test_loader=None,
                print_every=100, n_epochs=10, device='cpu'):
    for e in range(1, n_epochs+1):
        t0 = time.perf_counter()
        e_loss = []
        for batch_num, (seq_attnmask_labels) in enumerate(tqdm(train_loader), start=1):
            # Clear gradients
            optimizer.zero_grad()

            #get the 3 input args for this batch
            seq, attn_mask, labels = seq_attnmask_labels

            # Converting these to cuda tensors
            seq, attn_mask, labels = seq.cuda(device), attn_mask.cuda(device), labels.cuda(device)

            # Obtaining the logits from the model
            logits = net(seq, attn_mask)

            # Computing loss
            loss = criterion(logits.squeeze(-1), labels.long())
            e_loss.append(loss.item())

            # Backpropagating the gradients
            loss.backward()

            # Clip gradient norms
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)

            # Optimization step
            optimizer.step()
            # scheduler.step()

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
    # Creating instances of training and validation set
    train_set = dataset.dataset(dataset_fname = config.train_fname,
                                model_name = config.MODEL_NAME,
                                max_len=config.MAX_SEQ_LEN,
                                sample_ratio = None,
                                is_lower=config.IS_LOWER)
    test_set = dataset.dataset(dataset_fname = config.test_fname,
                               model_name=config.MODEL_NAME,
                               max_len=config.MAX_SEQ_LEN,
                               sample_ratio = None,
                               is_lower=config.IS_LOWER)

    #creating dataloader
    train_loader = DataLoader(train_set, shuffle = True,
                              batch_size=config.BATCH_SIZE,
                              num_workers=config.NUM_CPU_WORKERS)
    test_loader = DataLoader(test_set, shuffle = False,
                             batch_size=config.BATCH_SIZE,
                             num_workers=config.NUM_CPU_WORKERS)

    #creating BERT model
    bert_model = model.bert_classifier(model_name=config.MODEL_NAME,
                                       context_vector_size=config.CONTEXT_VECTOR_SIZE,
                                       freeze_bert=config.BERT_LAYER_FREEZE)
    bert_model.cuda()
    print(f"created NEW TRANSFORMER model for finetuning: {bert_model}")

    #loss function
    criterion = nn.CrossEntropyLoss()
    # criterion = nn.NLLLoss()

    #optimizer and scheduler
    optimizer = optim.Adam(bert_model.parameters(), lr=config.LR)
    scheduler = None

    # param_optimizer = list(bert_model.named_parameters())
    # no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    # optimizer_parameters = [
    #     {
    #         "params": [
    #             p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
    #         ],
    #         "weight_decay": 0.001,
    #     },
    #     {
    #         "params": [
    #             p for n, p in param_optimizer if any(nd in n for nd in no_decay)
    #         ],
    #         "weight_decay": 0.0,
    #     },
    # ]
    # num_train_steps = int(len(train_set) / config.BATCH_SIZE * config.NUM_EPOCHS)
    # optimizer = AdamW(optimizer_parameters, lr=config.LR)
    # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_train_steps)


    # Multi GPU setting
    if config.MULTIGPU:
        device_ids = [0, 1, 2, 3]  # huggingface allows parallelizing only upto 4 cards
        bert_model = nn.DataParallel(bert_model,  device_ids=device_ids)
        print(f'Model parallelized on the following cards: ', device_ids)

    train_model(bert_model, criterion, optimizer, scheduler, train_loader, test_loader,
                print_every=config.PRINT_EVERY, n_epochs=config.NUM_EPOCHS, device=config.DEVICE)


if __name__ == '__main__':
    main()