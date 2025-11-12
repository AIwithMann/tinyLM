import torch
from model import ChatBot
from training_loop import test_loader, train_dataset
from tqdm import tqdm
import math

# initialize model with same hyperparams
model = ChatBot(256, 4, 0.2, train_dataset.vocab_size, 128, 4, "gelu")

# load checkpoint correctly
checkpoint = torch.load("/home/username/Forever-Beta/checkpoints/best_val_loss_epoch_1.pt", map_location="cuda")
model.load_state_dict(checkpoint["model_state_dict"])
model.to('cuda')
model.eval()

loss_fn = torch.nn.CrossEntropyLoss()

total_loss = 0.0
total_correct = 0
total_tokens = 0

# proper no_grad context
with torch.no_grad():
    progress_bar = tqdm(test_loader, desc="[Test]", leave=False)
    for inputs, targets in progress_bar:
        inputs = inputs.to("cuda")
        targets = targets.to("cuda")

        outputs = model(inputs)
        loss = loss_fn(outputs.view(-1, train_dataset.vocab_size), targets.view(-1))
        total_loss += loss.item()

        preds = torch.argmax(outputs, dim=-1)
        correct = (preds == targets).sum().item()
        total = targets.numel()

        total_correct += correct
        total_tokens += total

        acc = total_correct / total_tokens
        progress_bar.set_postfix(loss=loss.item(), acc=acc)

test_acc = total_correct / total_tokens
test_loss = total_loss / len(test_loader)
test_ppl = math.exp(test_loss)
print(f"Test Accuracy: {test_acc:.4f}, Test Loss: {test_loss:.4f}, Perplexity: {test_ppl}")