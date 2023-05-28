from transformers import BertTokenizerFast, BertForSequenceClassification, Trainer, TrainingArguments, \
    BertForTokenClassification, DistilBertForTokenClassification, DistilBertTokenizerFast, RobertaTokenizerFast, \
    RobertaForTokenClassification, ElectraForTokenClassification, ElectraTokenizerFast
from torch.utils.data import Dataset, random_split
import torch
print(torch.version)
print(torch.cuda.is_available())
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(device)


model = ElectraForTokenClassification.from_pretrained('./my_model_electra')
tokenizer = ElectraTokenizerFast.from_pretrained('./my_tokenizer_electra')
model.to(device)  # Don't forget to move your model to the GPU if available
def predict(sentence):
    inputs = tokenizer(sentence, truncation=True, padding=True, return_tensors="pt")
    inputs = {name: tensor.to(device) for name, tensor in inputs.items()}  # Move the inputs to the GPU
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_senses = torch.argmax(logits, dim=-1).cpu().numpy().tolist()  # Take the argmax to get the most likely label
    return predicted_senses

# Now you can use the function to get predictions
sentence = "Es gribu ēst"
predicted_senses = predict(sentence)
print(f"The predicted senses in '{sentence}' are: '{predicted_senses}'")

