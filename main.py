#This code will train model and then predict the values party_type, Prty_name and memo
import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertPreTrainedModel, BertModel, Trainer, TrainingArguments
from datasets import Dataset
import torch
import torch.nn as nn

# Load and preprocess data
df = pd.read_csv('Transaction.csv', low_memory=False)

# Ensure REMARKS is a string
df['REMARKS'] = df['REMARKS'].astype(str)

# Filter the data where 'map_unmap' is 1
df = df[df['map_unmap'] == 1]

# Fill missing values
df['party_name'] = df['party_name'].fillna('unknown')
df['party_type'] = df['party_type'].fillna(3).astype(int)  # Fill missing with 3 and ensure it's int
df['memo'] = df['memo'].fillna('unknown')

# Convert categorical text labels to numerical labels
party_names_list = df['party_name'].unique().tolist()
memo_list = df['memo'].unique().tolist()

df['party_name'] = df['party_name'].apply(lambda x: party_names_list.index(x))
df['memo'] = df['memo'].apply(lambda x: memo_list.index(x))

# Save party_names_list and memo_list to JSON files for future use
with open('party_names_list.json', 'w') as f:
    json.dump(party_names_list, f)

with open('memo_list.json', 'w') as f:
    json.dump(memo_list, f)

# Split the data into training and evaluation sets
train_df, eval_df = train_test_split(df, test_size=0.2, random_state=42)

# Convert to Hugging Face dataset format
train_dataset = Dataset.from_pandas(train_df[['REMARKS', 'party_type', 'party_name', 'memo']])
eval_dataset = Dataset.from_pandas(eval_df[['REMARKS', 'party_type', 'party_name', 'memo']])

# Initialize tokenizer
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)

# Custom Multi-Task BERT Model
class MultiTaskBERT(BertPreTrainedModel):
    def __init__(self, config, num_labels_party_type=4, num_labels_party_name=len(party_names_list), num_labels_memo=len(memo_list)):
        super().__init__(config)
        self.num_labels_party_type = num_labels_party_type
        self.num_labels_party_name = num_labels_party_name
        self.num_labels_memo = num_labels_memo

        self.bert = BertModel(config)

        # Classification heads for each task
        self.party_type_classifier = nn.Linear(config.hidden_size, self.num_labels_party_type)
        self.party_name_classifier = nn.Linear(config.hidden_size, self.num_labels_party_name)
        self.memo_classifier = nn.Linear(config.hidden_size, self.num_labels_memo)

        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels_party_type=None, labels_party_name=None, labels_memo=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        pooled_output = outputs[1]  # Pooled output from BERT

        # Output heads
        party_type_logits = self.party_type_classifier(pooled_output)
        party_name_logits = self.party_name_classifier(pooled_output)
        memo_logits = self.memo_classifier(pooled_output)

        loss = None
        if labels_party_type is not None and labels_party_name is not None and labels_memo is not None:
            loss_fct = nn.CrossEntropyLoss()
            party_type_loss = loss_fct(party_type_logits.view(-1, self.num_labels_party_type), labels_party_type.view(-1))
            party_name_loss = loss_fct(party_name_logits.view(-1, self.num_labels_party_name), labels_party_name.view(-1))
            memo_loss = loss_fct(memo_logits.view(-1, self.num_labels_memo), labels_memo.view(-1))
            loss = party_type_loss + party_name_loss + memo_loss

        return (loss, party_type_logits, party_name_logits, memo_logits)


# Initialize the model
model = MultiTaskBERT.from_pretrained(model_name)

# Tokenization function
def preprocess_function(examples):
    # Tokenize the input text
    tokenized_inputs = tokenizer(examples['REMARKS'], padding='max_length', truncation=True)

    # Add labels for each task
    tokenized_inputs['labels_party_type'] = examples['party_type']
    tokenized_inputs['labels_party_name'] = examples['party_name']
    tokenized_inputs['labels_memo'] = examples['memo']

    return tokenized_inputs

# Apply tokenization
train_dataset = train_dataset.map(preprocess_function, batched=True)
eval_dataset = eval_dataset.map(preprocess_function, batched=True)

# Set format for PyTorch
train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels_party_type', 'labels_party_name', 'labels_memo'])
eval_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels_party_type', 'labels_party_name', 'labels_memo'])

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=10,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    eval_strategy="epoch",
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
)

# Train the model
trainer.train()

# Save the model and tokenizer
model.save_pretrained('./model')
tokenizer.save_pretrained('./model')

# Load the model, tokenizer, and saved lists for inference
model = MultiTaskBERT.from_pretrained('./model')
tokenizer = BertTokenizer.from_pretrained('./model')

with open('party_names_list.json', 'r') as f:
    party_names_list = json.load(f)

with open('memo_list.json', 'r') as f:
    memo_list = json.load(f)

# Inference function using loaded lists
def predict(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)

    party_type_logits, party_name_logits, memo_logits = outputs[1:]

    predicted_party_type_id = torch.argmax(party_type_logits, dim=1).item()
    predicted_party_name_id = torch.argmax(party_name_logits, dim=1).item()
    predicted_memo_id = torch.argmax(memo_logits, dim=1).item()

    predicted_party_name = party_names_list[predicted_party_name_id]  # Decode the party_name
    predicted_memo = memo_list[predicted_memo_id]  # Decode the memo

    return {
        'party_type': predicted_party_type_id,
        'party_name': predicted_party_name,
        'memo': predicted_memo
    }

# Example usage
text = "UPI-STONEBOLT ENTERPRISE-PAYTM-68779856@ PAYTM-PYTM0123456-362946011634-OID202309 202242120 362946011634"
predictions = predict(text, model, tokenizer)
print(predictions)

import pandas as pd
import torch
from torch import nn
from transformers import BertTokenizer, BertPreTrainedModel, BertModel
import json

# Load your model, tokenizer, and the lists
model_dir = './model'

# Load the party_name and memo lists
with open(f'party_names_list.json', 'r') as f:
    party_names_list = json.load(f)

with open(f'memo_list.json', 'r') as f:
    memo_list = json.load(f)

# Custom Multi-Task BERT Model
class MultiTaskBERT(BertPreTrainedModel):
    def __init__(self, config, num_labels_party_type=4, num_labels_party_name=len(party_names_list), num_labels_memo=len(memo_list)):
        super().__init__(config)
        self.num_labels_party_type = num_labels_party_type
        self.num_labels_party_name = num_labels_party_name
        self.num_labels_memo = num_labels_memo

        self.bert = BertModel(config)

        # Classification heads for each task
        self.party_type_classifier = nn.Linear(config.hidden_size, self.num_labels_party_type)
        self.party_name_classifier = nn.Linear(config.hidden_size, self.num_labels_party_name)
        self.memo_classifier = nn.Linear(config.hidden_size, self.num_labels_memo)

        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels_party_type=None, labels_party_name=None, labels_memo=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        pooled_output = outputs[1]  # Pooled output from BERT

        # Output heads
        party_type_logits = self.party_type_classifier(pooled_output)
        party_name_logits = self.party_name_classifier(pooled_output)
        memo_logits = self.memo_classifier(pooled_output)

        loss = None
        if labels_party_type is not None and labels_party_name is not None and labels_memo is not None:
            loss_fct = nn.CrossEntropyLoss()
            party_type_loss = loss_fct(party_type_logits.view(-1, self.num_labels_party_type), labels_party_type.view(-1))
            party_name_loss = loss_fct(party_name_logits.view(-1, self.num_labels_party_name), labels_party_name.view(-1))
            memo_loss = loss_fct(memo_logits.view(-1, self.num_labels_memo), labels_memo.view(-1))
            loss = party_type_loss + party_name_loss + memo_loss

        return (loss, party_type_logits, party_name_logits, memo_logits)

# Load the model and tokenizer
model = MultiTaskBERT.from_pretrained(model_dir)
tokenizer = BertTokenizer.from_pretrained(model_dir)

# Inference function with fallback for unknown memo
def predict(text, model, tokenizer, fallback_memo='Miscellaneous'):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)

    party_type_logits, party_name_logits, memo_logits = outputs[1:]

    predicted_party_type_id = torch.argmax(party_type_logits, dim=1).item()
    predicted_party_name_id = torch.argmax(party_name_logits, dim=1).item()
    predicted_memo_id = torch.argmax(memo_logits, dim=1).item()

    predicted_party_name = party_names_list[predicted_party_name_id]  # Decode the party_name
    predicted_memo = memo_list[predicted_memo_id]  # Decode the memo

    # Apply fallback for unknown memo
    if predicted_memo.lower() == 'unknown':
        predicted_memo = fallback_memo

    return {
        'party_type': predicted_party_type_id,
        'party_name': predicted_party_name,
        'memo': predicted_memo
    }

# Load the data from Bank_Import.csv
df = pd.read_csv('Bank_Import.csv')

# Apply the prediction function to each row in the REMARKS column
predictions = df['REMARKS'].apply(lambda x: predict(x, model, tokenizer))

# Create a new DataFrame with the predictions
pred_df = pd.DataFrame(predictions.tolist())

# Concatenate the original DataFrame with the predictions
result_df = pd.concat([df, pred_df], axis=1)

# Save the result to a new CSV file
result_df.to_csv('Bank_Import_Predictions1.csv', index=False)

print("Predictions saved to Bank_Import_Predictions1.csv")

