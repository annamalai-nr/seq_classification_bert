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
        self.fc = nn.Linear(context_vector_size, 2)

        #Dropout
        self.dp20 = nn.Dropout(0.2)

    def forward_bert(self, seq, attn_masks):
        '''
        Inputs:
            -seq : Tensor of shape [B, T] containing token ids of sequences
            -attn_masks : Tensor of shape [B, T] containing attention masks to be used to avoid contibution of PAD tokens
        '''

        # Feeding the input to BERT model to obtain contextualized representations
        last_hidden_states, pooled_op = self.bert_layer(seq, attention_mask=attn_masks)
        # del pooled_op

        # Obtaining the representation of [CLS] head
        # op = last_hidden_states.mean(1) #mean of all last hidden states
        op = pooled_op
        op = self.dp20(op)

        # Feeding cls_rep to the classifier layer
        logits = self.fc(op)


        return logits


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
        cls_rep = cont_reps[0].mean(1)
        cls_rep = self.dp20(cls_rep)

        # Feeding cls_rep to the classifier layer
        logits = self.fc(cls_rep)

        return logits