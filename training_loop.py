from torch.utils.data import Dataset, DataLoader
import pandas as pd
import model
import tokenizer
from tokenizers import Tokenizer
import torch
from tqdm import tqdm
import os

class chat_dataset(Dataset):
    def __init__(self, csv_file, tokenizer_file, max_len=128):
        self.data = pd.read_csv(csv_file)
        self.tokenizer = Tokenizer.from_file(tokenizer_file)
        self.max_len = max_len
        self.vocab_size = self.tokenizer.get_vocab_size()
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        p = row['prompt']
        r = row['response']

        p_ids = self.tokenizer.encode(p).ids
        r_ids = self.tokenizer.encode(r).ids

        p_ids = p_ids[:self.max_len]
        r_ids = r_ids[:self.max_len]

        p_ids += [0] * (self.max_len - len(p_ids))
        r_ids += [0] * (self.max_len - len(r_ids))

        return torch.tensor(p_ids, dtype=torch.long), torch.tensor(r_ids, dtype=torch.long)

# datasets
train_dataset = chat_dataset(
    csv_file="processsed_data/train_processed.csv",
    tokenizer_file="tokenizer.json"
)

validation_dataset = chat_dataset(
    csv_file="processed_data/validation_processed.csv",
    tokenizer_file="tokenizer.json"
)

test_dataset = chat_dataset(
    csv_file="processed_data/test_processed.csv",
    tokenizer_file="tokenizer.json"
)

# dataloaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2, pin_memory=True)
validation_loader = DataLoader(validation_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# model setup
chat_model = model.ChatBot(256, 4, 0.2, train_dataset.vocab_size, 128, 4, "gelu")
chat_model.to('cuda')
adam = torch.optim.AdamW(chat_model.parameters())
loss_fn = torch.nn.CrossEntropyLoss()

print(torch.cuda.is_available())

# checkpoint setup
best_val_loss = float('inf')
best_val_acc = 0.0
save_dir = "checkpoints"
os.makedirs(save_dir, exist_ok=True)

for epoch in range(15):
    # ---- Training ----
    chat_model.train()
    running_loss = 0.0
    total_correct = 0
    total_tokens = 0

    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]", leave=False)
    for inputs, targets in progress_bar:
        inputs, targets = inputs.to('cuda'), targets.to('cuda')
        adam.zero_grad()
        outputs = chat_model.forward(inputs)
        loss = loss_fn(outputs.view(-1, train_dataset.vocab_size), targets.view(-1))
        loss.backward()
        adam.step()

        running_loss = loss.item()
        preds = torch.argmax(outputs, dim=-1)
        correct = (preds == targets).sum().item()
        total = targets.numel()
        total_correct += correct
        total_tokens += total
        acc = total_correct / total_tokens
        progress_bar.set_postfix(loss=running_loss, acc=acc)

    train_acc = total_correct / total_tokens

    # ---- Validation ----
    chat_model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        val_bar = tqdm(validation_loader, desc=f"Epoch {epoch+1} [Val]", leave=False)
        for inputs, targets in val_bar:
            inputs, targets = inputs.to('cuda'), targets.to('cuda')
            outputs = chat_model.forward(inputs)
            loss = loss_fn(outputs.view(-1, train_dataset.vocab_size), targets.view(-1))
            val_loss += loss.item()
            preds = torch.argmax(outputs, dim=-1)
            val_correct += (preds == targets).sum().item()
            val_total += targets.numel()

    val_acc = val_correct / val_total
    val_loss /= len(validation_loader)

    print(f"Epoch {epoch+1} | Train Loss: {running_loss:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

    # ---- Checkpoint saving ----
    save_path = None
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        save_path = os.path.join(save_dir, f"best_val_loss_epoch_{epoch+1}.pt")
    elif val_acc > best_val_acc:
        best_val_acc = val_acc
        save_path = os.path.join(save_dir, f"best_val_acc_epoch_{epoch+1}.pt")

    if save_path:
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': chat_model.state_dict(),
            'optimizer_state_dict': adam.state_dict(),
            'val_loss': val_loss,
            'val_acc': val_acc
        }, save_path)
        print(f"Model saved: {save_path}")
