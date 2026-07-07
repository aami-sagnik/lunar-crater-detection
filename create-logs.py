import re
import matplotlib.pyplot as plt

model_name = "RetinaNet"
model_display_name = model_name
model_name = model_name.replace(" ", "-").lower()
log_data = ""
with open(f"./logs/{model_name}-log.txt", "r") as f:
    log_data = f.read()

epochs = []
train_losses = []
val_losses = []

pattern = r"Epoch\s+\[(\d+)/\d+\]\s+Train Loss:\s+([\d\.]+)\s+\|\s+Val Loss:\s+([\d\.]+)"
matches = re.findall(pattern, log_data)

for match in matches:
    epochs.append(int(match[0]))
    train_losses.append(float(match[1]))
    val_losses.append(float(match[2]))

print("Epochs:", epochs)
print("Train Losses:", train_losses)
print("Val Losses:", val_losses)

fig, ax = plt.subplots()
ax.plot(epochs, train_losses, label='Train Loss')
ax.plot(epochs, val_losses, label='Val Loss')
ax.set_title(model_display_name)
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.legend()
plt.savefig(f'./logs/{model_name}.jpg')
print("Saved successfully")