from sql_lvnet import return_all
from process_corpus_h import process_corpus
from generate_from_db import generate_data

sentences, labels = generate_data()
#print(f"Number of sentences: {len(sentences)}")
#print(sentences[0:3])

#print(f"Number of labels: {len(labels)}")
#print(labels[0:3])
'''
print(labels)

for i in range(len(sentences)):
    print("Original Sentence:", sentences[i])
    print("Replaced Sentence:", labels[i])
    print()
'''
max_value = max(max(sublist) for sublist in labels)
print(max_value)

'''
words, labels_corpus = process_corpus("princis.conll")
print(len(words), words)
print(len(labels_corpus), labels_corpus)
'''

from transformers import BertTokenizerFast, BertForSequenceClassification, Trainer, TrainingArguments, \
    BertForTokenClassification, DistilBertForTokenClassification, DistilBertTokenizerFast, RobertaTokenizerFast, \
    RobertaForTokenClassification, ElectraTokenizerFast, ElectraForTokenClassification
from torch.utils.data import Dataset, random_split
import torch

print(torch.version)
print(torch.cuda.is_available())
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(device)


class WSDDataSet(Dataset):
    def __init__(self, data, label):
        self.tokenizer = ElectraTokenizerFast.from_pretrained('google/electra-base-discriminator')
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sentence = self.data[idx]
        labels = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            sentence,
            truncation=True,
            padding='max_length',
            max_length=128,
            return_tensors='pt',
            return_attention_mask=True,
            return_token_type_ids=False
        )
        #print(sentence)
        #print(encoding['input_ids'].flatten())
        word_ids = encoding.word_ids()
        processed_labels = []
        for i, word_id in enumerate(word_ids):
            if word_id is None:
                # Special tokens have a word_id of None. The mask will automatically be set to -100 by PyTorch
                processed_labels.append(-100)
            else:
                if word_id < len(labels):
                    # For other tokens, set the label to the label of the word it came from
                    processed_labels.append(labels[word_id])
                else:
                    processed_labels.append(-100)

        #print(processed_labels)
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(processed_labels)
        }


wsd_dataset = WSDDataSet(sentences, labels)

# Let's split the dataset into train and test sets
train_size = int(0.8 * len(wsd_dataset))  # 80% for training
test_size = len(wsd_dataset) - train_size

train_dataset, test_dataset = random_split(wsd_dataset, [train_size, test_size])

model = ElectraForTokenClassification.from_pretrained('google/electra-base-discriminator', num_labels=max_value + 1)
model.to(device)  # Move the model to the GPU

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=True,
    fp16=True,
    warmup_steps=500,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)

trainer.train()
model.save_pretrained('./my_model_electra')
wsd_dataset.tokenizer.save_pretrained('./my_tokenizer_electra')

