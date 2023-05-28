from transformers import BertTokenizerFast, BertForSequenceClassification, Trainer, TrainingArguments, \
    BertForTokenClassification, DistilBertForTokenClassification, DistilBertTokenizerFast, \
    RobertaForTokenClassification, RobertaTokenizerFast, ElectraForTokenClassification, ElectraTokenizerFast
from torch.utils.data import Dataset, random_split
import torch
from sklearn.metrics import classification_report

def remove_duplicates(lst):
    result = []
    prev = None
    for i, curr in enumerate(lst):
        if curr != prev:
            result.append(curr)
        elif i > 1 and i < len(lst) - 1 and curr != lst[i + 1]:
            result.pop()  # Remove the first occurrence
            result.append(curr)  # Add the last occurrence
        prev = curr
    return result


def flatten_2d_array(nested_array):
    flattened_array = []
    for sublist in nested_array:
        for element in sublist:
            flattened_array.append(element)
    return flattened_array


from process_corpus_h import process_corpus

print(torch.version)
print(torch.cuda.is_available())
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(device)

model = BertForTokenClassification.from_pretrained('./my_model')
tokenizer = BertTokenizerFast.from_pretrained('./my_tokenizer')
model.to(device)  # Don't forget to move your model to the GPU if available


def predict(sentence):
    inputs = tokenizer(sentence, truncation=True, padding=True, return_tensors="pt")
    inputs = {name: tensor.to(device) for name, tensor in inputs.items()}  # Move the inputs to the GPU
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_senses = torch.argmax(logits,
                                    dim=-1).cpu().numpy().tolist()  # Take the argmax to get the most likely label
    return predicted_senses


# Now you can use the function to get predictions
words, labels_corpus = process_corpus("princis.conll")
#print(len(words), words)
#print(len(labels_corpus), labels_corpus)
# Preprocess the lists to divide them into sentences
new_s = []
new_l = []
tmp_s = []
tmp_l = []

for word_list, label_list in zip(words, labels_corpus):
    for w, l in zip(word_list, label_list):
        if l == -100:
            if tmp_s:  # Append the sentence if it's not empty
                new_s.append(' '.join(tmp_s))
                new_l.append(tmp_l)
                tmp_s = []  # Reset temporary lists
                tmp_l = []
            continue
        elif w in ('.', '!', '?'):
            tmp_s.append(w)
            tmp_l.append(l)
            new_s.append(' '.join(tmp_s))
            new_l.append(tmp_l)
            tmp_s = []  # Reset temporary lists
            tmp_l = []
        else:
            tmp_s.append(w)
            tmp_l.append(l)
win = 0
lose = 0
y_true_total = []
y_pred_total = []
for it, sen in enumerate(new_s):
    senses = new_l[it]
    predicted_senses = flatten_2d_array(predict(sen))
    predicted_senses = predicted_senses[1:-1]
    predicted_senses = remove_duplicates(predicted_senses)
    y_true = senses
    y_pred = predicted_senses[0:len(y_true)]
    if len(y_true) == len(y_pred):
        y_true_total.append(y_true)
        y_pred_total.append(y_pred)
        for i, s in enumerate(senses):
            if i < len(predicted_senses) and s == predicted_senses[i]:
                win += 1
            else:
                lose += 1
print(classification_report(flatten_2d_array(y_true_total), flatten_2d_array(y_pred_total)))
print(f'accuracy {win/lose*100}%')