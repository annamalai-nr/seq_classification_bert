import torch.nn as nn
from transformers import AutoModel


class bert_classifier(nn.Module):
    def __init__(self, model_name, context_vector_size, freeze_bert=True):
        super(bert_classifier, self).__init__()
        # Instantiating BERT model object
        self.bert_layer = AutoModel.from_pretrained(model_name)

        # Freeze bert layers
        if freeze_bert:
            for p in self.bert_layer.parameters():
                p.requires_grad = False

        # Classification layer
        self.fc1 = nn.Linear(context_vector_size, 100)
        self.fc2 = nn.Linear(100, 2)

        #Dropout
        self.dp20 = nn.Dropout(0.2)

        #RELU function
        self.relu = nn.ReLU()
        self.log_softmax  = nn.LogSoftmax(dim=1)


    def forward(self, seq, attn_masks):
        '''
        Inputs:
            -seq : Tensor of shape [B, T] containing token ids of sequences
            -attn_masks : Tensor of shape [B, T] containing attention masks to be used to avoid contibution of PAD tokens
        '''

        # Feeding the input to DistilBERT model to obtain contextualized representations
        cont_reps = self.bert_layer(seq, attention_mask=attn_masks)

        # Obtaining the representation of [CLS] head
        # cls_rep = cont_reps[0][:,0,:]
        op = cont_reps[0].mean(1)
        op = self.dp20(op)

        # Feeding op to the classifier layer
        op = self.relu(self.fc1(op))
        op = self.dp20(op)
        logits = self.fc2(op)

        logits = self.log_softmax(logits)

        return logits