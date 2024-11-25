
from torch.utils.data.dataset import random_split
from torchtext.data.functional import to_map_style_dataset
import pandas as pd

from torch.utils.data.dataset import random_split
from torchtext.data.functional import to_map_style_dataset

from torch import nn
import torch
import time
from torchtext.datasets import AG_NEWS
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator


#ortalocker is a Python library that is commonly used for creating file-level locks. It gives you the ability to lock files so that multiple processes or threads don't try to write to the same file simultaneously, which can prevent data corruption



# Create an iterator for the AG_NEWS dataset (split="train")
train_iter = iter(AG_NEWS(split="train"))

# Retrieve the first item from the dataset
first_item = next(train_iter)
print(f"First item - \n{first_item}")

# Retrieve the second item from the dataset
second_item = next(train_iter)
print(f"\nSecond item - \n{second_item}")


# Calling the training set
train_iter = AG_NEWS(split="train")

# Initialize a basic english tokenizer (converts to lowercase & splits by space/punctuation)
# Example tokenizing: "Hello, my name is Tamir!" --> ["hello", "my", "name", "is", "tamir"]
tokenizer = get_tokenizer("basic_english")


#The yield keyword in Python is used to define a generator function. When this function is called, it returns a generator object without beginning execution of the function. When next() is called for the first time, the function starts executing until it reaches the yield statement, which pauses the function and sends a value back to the caller. The function can then be resumed from where it left off by calling next() again.
def yield_tokens(data_iter):
    for _, text in data_iter:
        yield tokenizer(text)



# Build a look-up table of vocabulary (covert words/tokens to indices)
vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<unk>"])
# Set default index for unknown words
vocab.set_default_index(vocab["<unk>"])


example_sen = ['here', 'is', 'an', 'example']
example_sen_indexing = vocab(['here', 'is', 'an', 'example'])

print(f"Example sentence:\n{example_sen}")
print(f"\nExample sentence indexing:\n{example_sen_indexing}")


text_pipeline = lambda x: vocab(tokenizer(x))
label_pipeline = lambda x: int(x) - 1


# Get the first item from the dataset

first_item = next(iter(train_iter))

# Apply the pipelines
processed_text = text_pipeline(first_item[1])
processed_label = label_pipeline(first_item[0])

print(f"Processed Text (Indices):\n{processed_text}")
print(f"\nProcessed Label (Integer):\n{processed_label}")

from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def collate_batch(batch):
    label_list, text_list, offsets = [], [], [0]
    for _label, _text in batch:
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


train_iter = AG_NEWS(split="train")
dataloader = DataLoader(
    train_iter, batch_size=8, shuffle=False, collate_fn=collate_batch
)




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


train_iter = AG_NEWS(split="train")
num_class = len(set([label for (label, text) in train_iter]))
vocab_size = len(vocab)
emsize = 64
model = TextClassificationModel(vocab_size, emsize, num_class).to(device)

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


# Hyperparameters
EPOCHS = 10  # epoch
LR = 5  # learning rate
BATCH_SIZE = 64  # batch size for training

# Define loss function, optimizer & learning rate
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.1)

# Create iteratables for train set and test set
train_iter, test_iter = AG_NEWS()
# Convert iterable-style datasets to map-style datasets to enable random sampling the maps inside the DataLoader
train_dataset = to_map_style_dataset(train_iter)
test_dataset = to_map_style_dataset(test_iter)
# calulate how many samples are 95% and split into train/val
num_train = int(len(train_dataset) * 0.95)
split_train_, split_valid_ = random_split(train_dataset, [num_train, len(train_dataset) - num_train])

# Create the dataloaders
train_dataloader = DataLoader(split_train_, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch)
valid_dataloader = DataLoader(split_valid_, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch)


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

best_accu = None
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

print("Checking the results of test dataset.")
accu_test = evaluate(test_dataloader)
print("test accuracy {:8.3f}".format(accu_test))


ag_news_label = {1: "World", 2: "Sports", 3: "Business", 4: "Sci/Tec"}


def predict(text, text_pipeline):
    with torch.no_grad():
        text = torch.tensor(text_pipeline(text))
        output = model(text, torch.tensor([0]))
        return output.argmax(1).item() + 1

# This example is sports new
ex_text_str = "MEMPHIS, Tenn. – Four days ago, Jon Rahm was \
    enduring the season’s worst weather conditions on Sunday at The \
    Open on his way to a closing 75 at Royal Portrush, which \
    considering the wind and the rain was a respectable showing. \
    Thursday’s first round at the WGC-FedEx St. Jude Invitational \
    was another story. With temperatures in the mid-80s and hardly any \
    wind, the Spaniard was 13 strokes better in a flawless round. \
    Thanks to his best putting performance on the PGA Tour, Rahm \
    finished with an 8-under 62 for a three-stroke lead, which \
    was even more impressive considering he’d never played the \
    front nine at TPC Southwind."

model = model.to("cpu")

print("This is a %s news" % ag_news_label[predict(ex_text_str, text_pipeline)])

