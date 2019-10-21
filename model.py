import pandas as pd
import copy
from sklearn.preprocessing import LabelEncoder
import numpy as np
from tqdm import tqdm_notebook, trange

from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert import BertForSequenceClassification
from pytorch_pretrained_bert.optimization import BertAdam
from torch import nn
import torch
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F

import sys
sys.path.append(r'path_of_download')
from model_utils import *

# create toy data with categorical, continous and text data
train_df = pd.DataFrame({'cat1' : ['1','2','1','3','4'], 'cont1' : [123,31,43,12,32],'cont2' : [12,145,55,12,2],
                      'text':['this it first','this it second','this is third','this it second','this is one'] ,'y' : ['one','two','three','two','one']})

test_df = pd.DataFrame({'cat1' : ['1','4'], 'cont1' : [100,20],'cont2' : [52, 13],
                        'text':['this it first','this is first'] ,'y' : ['one','one']})

labels = ['one','two','three']

# declare some variables
TRAIN_BATCH_SIZE = 5
EVAL_BATCH_SIZE = 1
LEARNING_RATE = 5e-5
NUM_TRAIN_EPOCHS = 10
RANDOM_SEED = 42
WARMUP_PROPORTION = 0.1
MAX_SEQ_LENGTH = 10
MODEL_BERT = 'bert-base-uncased'   

tokenizer = BertTokenizer.from_pretrained(MODEL_BERT, do_lower_case=True)

# data format according to Bert model 
train_df_bert = pd.DataFrame({
    'id':range(len(train_df)),
    'label':train_df['y'],
    'alpha':['a']*train_df.shape[0],
    'text': train_df['text']
})

test_df_bert = pd.DataFrame({
    'id':range(len(test_df)),
    'label':test_df['y'],
    'alpha':['a']*test_df.shape[0],
    'text': test_df['text']
})

train_df_bert.to_csv('train.tsv', sep='\t', index=False, header=False)
test_df_bert.to_csv('test.tsv', sep='\t', index=False, header=False)

y_train = train_df['y']
generator = DataGenerator(labels, './', "train.tsv")
train = generator.get_examples("train")
train_examples_len = len(train)
num_labels = len(labels)


label_map = {str(label): i for i, label in enumerate(labels)}
train_for_processing = [(example, label_map, MAX_SEQ_LENGTH, tokenizer) for example in train]


train_features = list(map(convert_example_to_feature, train_for_processing))

generator_test = DataGenerator(labels, './', "test.tsv")
test = generator_test.get_examples("test")
test_examples_len = len(test)

test_for_processing = [(example, label_map, MAX_SEQ_LENGTH, tokenizer) for example in test]
test_features = list(map(convert_example_to_feature, test_for_processing))


class Bert_Model(nn.Module):

    def __init__(self, num_labels=2, emb_dropout=0.1, cont_vars_droput=0.05, feed_dict_cat={}, cont_vars=[]):
        super().__init__()
        self.num_labels = num_labels
        self.bert = BertForSequenceClassification.from_pretrained(MODEL_BERT, num_labels=num_labels)

        # replace classifier with a hidden layer
        self.bert.classifier = nn.Linear(self.bert.config.hidden_size, self.bert.config.hidden_size)
        
        # append to dict embedding layers for each categorical
        self.emb_layers = {}
        for key, val in feed_dict_cat.items():
            self.emb_layers[key] = nn.Embedding(val[0], val[1])  # (size of the dictionary of embeddings, the size of each embedding vector)

        self.n_cont = len(cont_vars)
        # append continous variables
        self.bn_layers = nn.BatchNorm1d(self.n_cont)  # normalize continous variables for an accelerated training
        self.cont_layers = nn.Linear(self.n_cont, self.bert.config.hidden_size)
        self.droput_cont = nn.Dropout(cont_vars_droput)
        nn.init.xavier_normal_(self.cont_layers.weight)

        # create linear layer after embedding
        no_of_embs = sum([val[1] for _, val in feed_dict_cat.items()])
        self.lin_emb = nn.Linear(no_of_embs, no_of_embs * 3)
        nn.init.xavier_normal_(self.lin_emb.weight)

        self.emb_dropout_layer = nn.Dropout(emb_dropout)
        
        # new final classifier, as input number of features from text embedding, categorical variables and continous variables
        self.classifier = nn.Linear(
            self.bert.classifier.out_features + self.lin_emb.out_features + self.cont_layers.out_features, num_labels)
        nn.init.xavier_normal_(self.bert.classifier.weight)

    def forward(self, cont_data, cat_data, input_ids, segment_ids=None, input_mask=None, labels=None):
        # Bert output
        pooled_output = self.bert(input_ids, segment_ids, input_mask, labels=None)

        # continous data
        x_cont = self.bn_layers(cont_data)
        x_cont = F.relu(self.cont_layers(x_cont))
        x_cont = self.droput_cont(x_cont)

        # Embeddings
        x_cat = [emb_layer(cat_data[key]) for key, emb_layer in self.emb_layers.items()]
        x_cat = torch.cat(x_cat, 1)
        x_cat = self.emb_dropout_layer(x_cat)
        emb_layer = self.lin_emb(x_cat)
        emb_layer = F.relu(emb_layer)
        emb_layer = self.emb_dropout_layer(emb_layer)

        # concatenate the output from bert, categorical embedding and continous variables
        concat = torch.cat([pooled_output, emb_layer, x_cont], 1)

        output_layer = self.classifier(concat)

        return output_layer



cat_vars = ['cat1']
cont_vars = ['cont1', 'cont2']

cat_sizes = {}
cat_embsizes = {}
for cat in cat_vars:
    cat_sizes[cat] = train_df[cat].nunique()
    cat_embsizes[cat] = min(50, cat_sizes[cat] // 2 + 1)

feed_dict = {cat: (cat_sizes[cat] + 1, cat_embsizes[cat]) for cat in cat_vars}

model_torch = Bert_Model(num_labels=len(label_map), feed_dict_cat=feed_dict, cont_vars=cont_vars)

# we can freeze the layers below to get a speed up in training without losing much accuracy
#cont = True
#for param in model_torch.named_parameters():
#    if param[0] ==  'bert.bert.encoder.layer.11.output.LayerNorm.bias':
#        cont = False
#    if cont: continue
#    param[1].requires_grad = False

param_optimizer = list(model_torch.named_parameters())
num_train_optimization_steps = int(train_examples_len / TRAIN_BATCH_SIZE) * NUM_TRAIN_EPOCHS


data_train = pd.DataFrame(index = range(len(train_df)))
data_test =  pd.DataFrame(index = range(len(test_df)))
label_encoders = {}
for cat_col in cat_vars:
    label_encoders[cat_col] = LabelEncoder()
    label_encoders[cat_col].fit(train_df[cat_col]) 
    data_train[cat_col] = label_encoders[cat_col].transform(train_df[cat_col])
    data_test[cat_col] = label_encoders[cat_col].transform(test_df[cat_col])

for cont_col in cont_vars:
    data_train[cont_col] = train_df[cont_col].values
    data_test[cont_col] = test_df[cont_col].values

loader_train = Load_Data(data_train, cat_vars, cont_vars, train_features)
dataloader_train = DataLoader(loader_train, batch_size=TRAIN_BATCH_SIZE)


loader_test = Load_Data(data_test, cat_vars, cont_vars, test_features)
dataloader_test = DataLoader(loader_test, batch_size=1)

# adding weights decay
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]


optimizer = BertAdam(optimizer_grouped_parameters,
                     lr=LEARNING_RATE,
                     warmup=WARMUP_PROPORTION,
                     t_total=num_train_optimization_steps)


loss_fn = CrossEntropyLoss()
losses = []
preds = []
train_step = make_train_step(model_torch, loss_fn, optimizer)
accuracy = 0.0

y_test = test_df.y.replace(label_map)

eval_loss = 0
nb_eval_steps = 0


for _ in trange(int(NUM_TRAIN_EPOCHS), desc="Epoch"):
    
    print('training \n')
    steps, loss = train_step(dataloader_train)
    losses.append(loss)
    print("training loss : \n")
    print('\n testing')
    for step, batch in enumerate(tqdm_notebook(dataloader_test, desc="Iteration")):
        
        input_ids, input_mask = batch['all_input_ids'], batch['all_input_mask']
        segment_ids, label_ids = batch['all_segment_ids'], batch['all_label_ids']
        cont_data = batch['cont_cols']
        cat_data = {key : batch[key] for key in batch.keys() if key not in ['all_input_ids','all_label_ids'
                                                                            'all_input_mask','all_segment_ids', 'cont_cols']}
        
        with torch.no_grad():
            _ = model_torch.eval()
            logits = model_torch.forward(cont_data, cat_data, input_ids, segment_ids, input_mask, labels=None)
        tmp_eval_loss = loss_fn(logits.view(-1, num_labels), label_ids.view(-1))

        eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
        
        preds.append(np.argmax(logits.detach().cpu().numpy(),axis=1)[0])
    
    eval_loss = eval_loss / nb_eval_steps
    
    epoch_accuracy = len(np.where(y_test.values == preds)[0]) / y_test.shape[0]
    print(' accuracy : ' , epoch_accuracy * 100)
    print(' eval loss : ', eval_loss)
    
    if accuracy < epoch_accuracy:
        print('current accuracy {}'.format(epoch_accuracy),
              'improved over previous {}'.format(accuracy))
        accuracy = epoch_accuracy
        best_model = copy.deepcopy(model_torch.state_dict())
        torch.save(model_torch.state_dict(), 'bert_model_best.pth')
        
    losses = []
    preds = []
    eval_loss = 0
    nb_eval_steps = 0

