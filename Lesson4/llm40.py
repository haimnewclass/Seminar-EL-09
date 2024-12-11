import torch
from datasets import load_dataset
from torch.nn.utils import prune
from torch.utils.data import DataLoader
from torch import nn
from torch.utils.data.dataset import random_split
from collections import defaultdict


#World News - Label 0
#Sports - Label 1
#Business - Label 2
#Science/Technology - Label 3

from transformers import AutoTokenizer

class MapStyleDataset:
    def __init__(self, iterator_dataset):
        self.data = list(iterator_dataset)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

def convert_to_map_style(iterator_dataset):
    return MapStyleDataset(iterator_dataset)




device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


dataset = load_dataset("ag_news")
train_iter = iter(dataset["train"])
test_iter = iter(dataset["test"])

# Retrieve the first item from the dataset
first_item = next(train_iter)
print(f"First item - \n{first_item}")

# Retrieve the second item from the dataset
second_item = next(train_iter)
print(f"\nSecond item - \n{second_item}")

# Initialize tokenizer (e.g., BERT)
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased",unk_token="<unk>")

unk_work = tokenizer.unk_token
unk_token_id = tokenizer.convert_tokens_to_ids(unk_work)

vocab = tokenizer.get_vocab()



def words_to_indices(words):
    """
    Convert an array of words to their corresponding token indices.

    Args:
    words (list of str): The list of words to convert.

    Returns:
    list of int: The list of token indices.
    """
    indices = []
    for word in words:
        tokens = tokenizer.tokenize(word)  # Tokenize the word
        token_ids = []
        for token in tokens:
            try:
                token_id = tokenizer.convert_tokens_to_ids(token)  # Try to get token ID
            except KeyError:
                token_id = tokenizer.convert_tokens_to_ids(tokenizer.unk_token)  # Get ID of unknown token
            token_ids.append(token_id)  # Append token ID
        if token_ids:
            indices.append(token_ids)  # Append IDs for this word
        else:
            unk_work = tokenizer.unk_token
            unk_token_id = tokenizer.convert_tokens_to_ids(unk_work)
            indices.append(unk_token_id)  # Append IDs for this word

        # Flattening the list using list comprehension
        flattened_list = [element for sublist in indices for element in
                          (sublist if isinstance(sublist, list) else [sublist])]


    return flattened_list




vocab_results = words_to_indices(['hello', 'world'])
# Word to check
word = "word"

# Check if the word exists
if word in vocab:
    print(f"'{word}' is in the vocabulary with ID: {vocab[word]}")
    wordToken = tokenizer.tokenize(word)
else:
    print(f"'{word}' is NOT in the vocabulary.")

def text_pipeline_func(x):
    return words_to_indices(x)

text_pipeline = text_pipeline_func
def label_pipeline_func(x):
    return int(x)

label_pipeline = label_pipeline_func


tokens = tokenizer("Hello world!")  # Returns input_ids, attention_mask
tokens = tokenizer(["Hello", "Hi"])  # Batch tokenization

processed_text = text_pipeline(first_item['text'])
processed_label = label_pipeline(first_item['label'])

tokens = tokenizer.encode("Hello")
text = tokenizer.decode(tokens)

def collate_batch(batch):
    label_list, text_list, offsets = [], [], [0]
    for item in batch:
        _label = item['label']
        _text = item['text']
        # Send each label of the batch through the label pipeline
        label_list.append(label_pipeline(_label))
        # Send each text of the batch through the text pipeline (and convert to tensor)
        processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
        text_list.append(processed_text)
        # 'offsets' keeps track of the beginning index of each individual sequence in the text tensor.
        offsets.append(processed_text.size(0))
    label_list = torch.tensor(label_list, dtype=torch.int64)
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    text_list = torch.cat(text_list)
    return label_list.to(device), text_list.to(device), offsets.to(device)


dataloader = DataLoader(train_iter,batch_size=8,shuffle=False,collate_fn=collate_batch)





class TextClassificationModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_class):
        super(TextClassificationModel, self).__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=False)
        self.fc = nn.Linear(embed_dim, num_class)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        fully_connected = self.fc(embedded)
        return fully_connected


num_class = 4 # len(set([label for (label, text) in train_iter]))
vocab_size = len(vocab)
emsize = 64
model = TextClassificationModel(vocab_size, emsize, num_class).to(device)

import time

# Hyperparameters
EPOCHS = 50  # epoch
LR = 5  # learning rate
BATCH_SIZE = 64  # batch size for training

# Define loss function, optimizer & learning rate
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.1)


train_iter = iter(dataset["train"])
test_iter = iter(dataset["test"])

# Convert iterable-style datasets to map-style datasets to enable random sampling the maps inside the DataLoader
train_dataset = convert_to_map_style(train_iter)
test_dataset = convert_to_map_style(test_iter)
# calulate how many samples are 95% and split into train/val
num_train = int(len(train_dataset) * 0.95)
split_train_, split_valid_ = random_split(train_dataset, [num_train, len(train_dataset) - num_train])
# Create the dataloaders
train_dataloader = DataLoader(split_train_, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch)
valid_dataloader = DataLoader(split_valid_, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch)

best_accu = None



def train(dataloader):
    model.train()
    total_acc, total_count = 0, 0
    log_interval = 500
    start_time = time.time()

    for idx, (label, text, offsets) in enumerate(dataloader):
        optimizer.zero_grad()
        predicted_label = model(text, offsets)
        loss = criterion(predicted_label, label)
        loss.backward()
        # Prevents exploding gradients by capping their magnitude
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()
        total_acc += (predicted_label.argmax(1) == label).sum().item()
        total_count += label.size(0)
        if idx % log_interval == 0 and idx > 0:
            elapsed = time.time() - start_time
            print(
                "| epoch {:3d} | {:5d}/{:5d} batches "
                "| accuracy {:8.3f}".format(
                    epoch, idx, len(dataloader), total_acc / total_count
                )
            )
            total_acc, total_count = 0, 0
            start_time = time.time()


def evaluate(dataloader):
    model.eval()
    total_acc, total_count = 0, 0

    with torch.no_grad():
        for idx, (label, text, offsets) in enumerate(dataloader):
            predicted_label = model(text, offsets)
            loss = criterion(predicted_label, label)
            total_acc += (predicted_label.argmax(1) == label).sum().item()
            total_count += label.size(0)
    return total_acc / total_count



def text_preprocess(text):
    # Tokenize the input text
    tokens_list = tokenizer(text)

    # Convert tokens to indices

    vectors = [vocab[token] for token in tokens_list if token in vocab]
    # Convert list of indices to tensor
    text_tensor = torch.tensor(vectors)

    # Generate a tensor of offsets
    offsets = torch.tensor([0])

    return text_tensor, offsets




def load_model_and_tokenizer(model_path='model.pth'):
    """
    Loads the pre-trained model and tokenizer.

    Args:
        model_path (str): Path to the saved model.

    Returns:
        tuple: Loaded model and tokenizer.
    """
    # Load the model state and tokenizer
    model = TextClassificationModel(vocab_size, emsize, num_class)
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", unk_token="<unk>")

    return model, tokenizer


# Prediction function
def predict_text_classification(text, model, tokenizer):
    """
    Predicts the class of the given text input.

    Args:
        text (str): The input text to classify.
        model: The pre-trained model.
        tokenizer: The tokenizer corresponding to the model.

    Returns:
        int: The predicted class label.
    """
    # Tokenize and preprocess the input text
    tokens = tokenizer.tokenize(text)
    indices = [tokenizer.convert_tokens_to_ids(token) for token in tokens]

    # Convert to tensor
    text_tensor = torch.tensor(indices, dtype=torch.int64).to(device)
    offsets = torch.tensor([0], dtype=torch.int64).to(device)

    # Run the model to get predictions
    with torch.no_grad():
        output = model(text_tensor, offsets)
        predicted_label = output.argmax(1).item()

    return predicted_label


# Example usage
def predict():
    model, tokenizer = load_model_and_tokenizer()
    example_text = "Liverpool all but qualified for the last 16 of the Champions League after Mohamed Salah's VAR-overturned penalty secured a narrow 1-0 win over Girona at the Municipal de Montilivi."
    predicted_class = predict_text_classification(example_text, model, tokenizer)
    print(f"Predicted class for the input text: {predicted_class}")

    example_text = "Crowning Achievement: NVIDIA Research Model Enables Fast, Efficient Dynamic Scene Reconstruction Content streaming and engagement are entering a new dimension with QUEEN, an AI model by NVIDIA Research and the University of Maryland that makes it possible to stream free-viewpoint video, which lets viewers experience a 3D scene from any angle"
    predicted_class = predict_text_classification(example_text, model, tokenizer)

    print(f"Predicted class for the input text: {predicted_class}")


def predict_pruing():
    model, tokenizer = load_model_and_tokenizer()
    example_text = "Liverpool all but qualified for the last 16 of the Champions League after Mohamed Salah's VAR-overturned penalty secured a narrow 1-0 win over Girona at the Municipal de Montilivi."
    example_text = "Crowning Achievement: NVIDIA Research Model Enables Fast, Efficient Dynamic Scene Reconstruction Content streaming and engagement are entering a new dimension with QUEEN, an AI model by NVIDIA Research and the University of Maryland that makes it possible to stream free-viewpoint video, which lets viewers experience a 3D scene from any angle"
    predicted_class = predict_text_classification(example_text, model, tokenizer)

    """
      Applies global unstructured pruning to the model.

      Args:
          model: The PyTorch model to prune.
          amount (float): The proportion of weights to prune globally.

      Returns:
          model: The pruned model.
      """
    parameters_to_prune = []
    for name, module in model.named_modules():
        print(f"Inspecting module: {name}, type: {type(module)}")
        if isinstance(module, (torch.nn.Linear, torch.nn.EmbeddingBag)):
            parameters_to_prune.append((module, 'weight'))

    if not parameters_to_prune:
        raise ValueError("No parameters found to prune. Check the model architecture or layer types.")

    print(f"Parameters to prune: {parameters_to_prune}")

    original_num_weights = sum(p[0].weight.nelement() for p in parameters_to_prune)
    original_num_nonzero = sum(p[0].weight.count_nonzero().item() for p in parameters_to_prune)

    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=0.2,
    )

    pruned_num_nonzero = sum(p[0].weight.count_nonzero().item() for p in parameters_to_prune)

    print(f"Original total weights: {original_num_weights}")
    print(f"Original nonzero weights: {original_num_nonzero}")
    print(f"Remaining nonzero weights after pruning: {pruned_num_nonzero}")
    print(f"Total weights pruned: {original_num_nonzero - pruned_num_nonzero}")
    print(f"Predicted class for the input text: {predicted_class}")


predict()
#predict_pruing()


for epoch in range(1, EPOCHS + 1):
    epoch_start_time = time.time()
    train(train_dataloader)
    accu_val = evaluate(valid_dataloader)
    if best_accu is not None and best_accu > accu_val:
        scheduler.step()
    else:
        best_accu = accu_val
    print("-" * 59)
    print(
        "| end of epoch {:3d} | time: {:5.2f}s | "
        "valid accuracy {:8.3f} ".format(
            epoch, time.time() - epoch_start_time, accu_val
        )
    )
    print("-" * 59)

torch.save(model.state_dict(), 'model_50.pth')

#model = TextClassificationModel(vocab_size, emsize, num_class)
#model.load_state_dict(torch.load('model.pth'))
#model.to(device)



import torch




