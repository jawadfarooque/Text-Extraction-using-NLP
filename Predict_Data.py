#This code will predict the values party_type, Prty_name and memo
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
model = MultiTaskBERT.from_pretrained(model_dir,ignore_mismatched_sizes=True)
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


