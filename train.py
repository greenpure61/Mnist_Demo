import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# -------------------------------------------------------------
# 🧩 MODELDEFINITION
# En simpel men effektiv CNN, der lærer at genkende håndskrevne cifre.
# Arkitekturen er inspireret af LeNet (fra 1998) – to convolution-lag
# til at finde mønstre + to fully connected lag til klassifikation.
# -------------------------------------------------------------
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            # Første convolution: finder simple features som kanter og streger
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),  # reducerer 28x28 -> 14x14

            # Anden convolution: finder mere komplekse mønstre (kombinationer af kanter)
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),  # reducerer 14x14 -> 7x7

            # Fladgør (flatten) til ét langt vektorrum til de fuldt forbundne lag
            nn.Flatten(),

            # Fully connected lag: klassisk MLP-toppen af CNN
            nn.Linear(64 * 7 * 7, 128), nn.ReLU(),

            # Sidste lag outputter 10 logits (ét pr. ciffer 0–9)
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.net(x)


# -------------------------------------------------------------
# 📦 DATA & LOADERS
# MNIST downloades automatisk via torchvision. Vi normaliserer
# billederne så de får mean=0.1307 og std=0.3081 (standard for MNIST).
# -------------------------------------------------------------
def get_loaders(batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_ds = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_ds  = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True),
        DataLoader(test_ds, batch_size=512, shuffle=False, num_workers=2, pin_memory=True)
    )


# -------------------------------------------------------------
# 🔁 TRÆNINGSPR. EPOKE
# Her sker forward -> loss -> backward -> opdatering.
# Vi holder styr på både loss og accuracy for at kunne måle forbedring.
# -------------------------------------------------------------
def train_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total, correct, running_loss = 0, 0, 0.0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()

        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * x.size(0)
        pred = logits.argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)

    return running_loss / total, correct / total


# -------------------------------------------------------------
# 🧪 EVALUERING (ingen gradienter)
# Vi bruger torch.no_grad() for at spare hukommelse og beregningstid.
# -------------------------------------------------------------
@torch.no_grad()
def evaluate(model, loader, loss_fn, device):
    model.eval()
    total, correct, running_loss = 0, 0, 0.0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = loss_fn(logits, y)

        running_loss += loss.item() * x.size(0)
        pred = logits.argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)

    return running_loss / total, correct / total


# -------------------------------------------------------------
# 🚀 MAIN LOOP
# Her samles hele træningen: vi vælger device, definerer model,
# loss, optimizer og gemmer den bedste model baseret på test-accuracy.
# -------------------------------------------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Træner på: {device}")

    train_loader, test_loader = get_loaders()
    model = CNN().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)

    best_acc = 0.0

    for epoch in range(1, 6):
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, loss_fn, device)
        te_loss, te_acc = evaluate(model, test_loader, loss_fn, device)

        print(f"Epoch {epoch}: train loss={tr_loss:.4f} acc={tr_acc:.4f} | "
              f"test loss={te_loss:.4f} acc={te_acc:.4f}")

        # Gem den bedste model (baseret på test-accuracy)
        if te_acc > best_acc:
            best_acc = te_acc
            torch.save({"state_dict": model.state_dict()}, "mnist_cnn.pt")

    print(f"\n✅ Træning færdig! Bedste test-accuracy: {best_acc:.4f}")
    print("Model gemt som 'mnist_cnn.pt'")

# -------------------------------------------------------------
# 🏁 KØRNING
# -------------------------------------------------------------
if __name__ == "__main__":
    main()
