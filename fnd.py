import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import transformers
from transformers import AutoModel, BertTokenizerFast

device = torch.device("cuda")

df = pd.read_csv("spamdata_v2.csv")
df.head()
df.shape
     
(5572, 2)

df['label'].value_counts(normalize = True)
     


train_text, temp_text, train_labels, temp_labels = train_test_split(df['text'], df['label'], 
                                                                    random_state=2018, 
                                                                    test_size=0.3, 
                                                                    stratify=df['label'])


val_text, test_text, val_labels, test_labels = train_test_split(temp_text, temp_labels, 
                                                                random_state=2018, 
                                                                test_size=0.5, 
                                                                stratify=temp_labels)
     



bert = AutoModel.from_pretrained('bert-base-uncased')


tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
     

text = ["this is a bert model tutorial", "we will fine-tune a bert model"]


sent_id = tokenizer.batch_encode_plus(text, padding=True, return_token_type_ids=False)
     


print(sent_id)
     
{'input_ids': [[101, 2023, 2003, 1037, 14324, 2944, 14924, 4818, 102, 0], [101, 2057, 2097, 2986, 1011, 8694, 1037, 14324, 2944, 102]], 'attention_mask': [[1, 1, 1, 1, 1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]}
Tokenization


seq_len = [len(i.split()) for i in train_text]

pd.Series(seq_len).hist(bins = 30)
     



max_seq_len = 25
     

tokens_train = tokenizer.batch_encode_plus(
    train_text.tolist(),
    max_length = max_seq_len,
    pad_to_max_length=True,
    truncation=True,
    return_token_type_ids=False
)


tokens_val = tokenizer.batch_encode_plus(
    val_text.tolist(),
    max_length = max_seq_len,
    pad_to_max_length=True,
    truncation=True,
    return_token_type_ids=False
)


tokens_test = tokenizer.batch_encode_plus(
    test_text.tolist(),
    max_length = max_seq_len,
    pad_to_max_length=True,
    truncation=True,
    return_token_type_ids=False
)
     


train_seq = torch.tensor(tokens_train['input_ids'])
train_mask = torch.tensor(tokens_train['attention_mask'])
train_y = torch.tensor(train_labels.tolist())


val_seq = torch.tensor(tokens_val['input_ids'])
val_mask = torch.tensor(tokens_val['attention_mask'])
val_y = torch.tensor(val_labels.tolist())


test_seq = torch.tensor(tokens_test['input_ids'])
test_mask = torch.tensor(tokens_test['attention_mask'])
test_y = torch.tensor(test_labels.tolist())
     

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler


batch_size = 32


train_data = TensorDataset(train_seq, train_mask, train_y)


train_sampler = RandomSampler(train_data)


train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)


val_data = TensorDataset(val_seq, val_mask, val_y)


val_sampler = SequentialSampler(val_data)


val_dataloader = DataLoader(val_data, sampler = val_sampler, batch_size=batch_size)
     

for param in bert.parameters():
    param.requires_grad = False
     


class BERT_Arch(nn.Module):

    def __init__(self, bert):
      
      super(BERT_Arch, self).__init__()

      self.bert = bert 
      
 
      self.dropout = nn.Dropout(0.1)
      
      
      self.relu =  nn.ReLU()

      
      self.fc1 = nn.Linear(768,512)
      
      
      self.fc2 = nn.Linear(512,2)

      
      self.softmax = nn.LogSoftmax(dim=1)

  
    def forward(self, sent_id, mask):

  
      _, cls_hs = self.bert(sent_id, attention_mask=mask)
      
      x = self.fc1(cls_hs)

      x = self.relu(x)

      x = self.dropout(x)

  
      x = self.fc2(x)
      
    
      x = self.softmax(x)

      return x
     


model = BERT_Arch(bert)


model = model.to(device)
     

from transformers import AdamW


optimizer = AdamW(model.parameters(), lr = 1e-3)
     

from sklearn.utils.class_weight import compute_class_weight


class_wts = compute_class_weight('balanced', np.unique(train_labels), train_labels)

print(class_wts)
     
[0.57743559 3.72848948]


weights= torch.tensor(class_wts,dtype=torch.float)
weights = weights.to(device)


cross_entropy  = nn.NLLLoss(weight=weights) 


epochs = 10
     

def train():
  
  model.train()

  total_loss, total_accuracy = 0, 0
  

  total_preds=[]
  

  for step,batch in enumerate(train_dataloader):
    
  
    if step % 50 == 0 and not step == 0:
      print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(train_dataloader)))


    batch = [r.to(device) for r in batch]
 
    sent_id, mask, labels = batch


    model.zero_grad()        

 
    preds = model(sent_id, mask)

 
    loss = cross_entropy(preds, labels)

 
    total_loss = total_loss + loss.item()

  
    loss.backward()

    
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    
    optimizer.step()

   
    preds=preds.detach().cpu().numpy()

   
    total_preds.append(preds)

  
  avg_loss = total_loss / len(train_dataloader)
  
 
  total_preds  = np.concatenate(total_preds, axis=0)

  
  return avg_loss, total_preds
     


def evaluate():
  
  print("\nEvaluating...")
  

  model.eval()

  total_loss, total_accuracy = 0, 0
  

  total_preds = []

\
  for step,batch in enumerate(val_dataloader):
    
    
    if step % 50 == 0 and not step == 0:
      

      elapsed = format_time(time.time() - t0)
            
  
      print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(val_dataloader)))

   
    batch = [t.to(device) for t in batch]

    sent_id, mask, labels = batch

   
    with torch.no_grad():
      
     
      preds = model(sent_id, mask)

     
      loss = cross_entropy(preds,labels)

      total_loss = total_loss + loss.item()

      preds = preds.detach().cpu().numpy()

      total_preds.append(preds)

  
  avg_loss = total_loss / len(val_dataloader) 

 
  total_preds  = np.concatenate(total_preds, axis=0)

  return avg_loss, total_preds
     
best_valid_loss = float('inf')


train_losses=[]
valid_losses=[]


for epoch in range(epochs):
     
    print('\n Epoch {:} / {:}'.format(epoch + 1, epochs))
    

    train_loss, _ = train()
    

    valid_loss, _ = evaluate()
    
  
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'saved_weights.pt')
    

    train_losses.append(train_loss)
    valid_losses.append(valid_loss)
    
    print(f'\nTraining Loss: {train_loss:.3f}')
    print(f'Validation Loss: {valid_loss:.3f}')


path = 'saved_weights.pt'
model.load_state_dict(torch.load(path))
     

with torch.no_grad():
  preds = model(test_seq.to(device), test_mask.to(device))
  preds = preds.detach().cpu().numpy()
     


preds = np.argmax(preds, axis = 1)
print(classification_report(test_y, preds))
     
           

pd.crosstab(test_y, preds)
     

