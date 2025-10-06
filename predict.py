"""
predict.py
-----------
CLI-værktøj til at lave én forudsigelse på et billede.
Eksempel:
    python predict.py path/til/billede.png
"""

import sys
import torch
import cv2
import numpy as np
from torchvision import transforms

from train import CNN  # genbrug samme arkitektur som ved træning


# -------------------------------------------------------------
# 🔧 PREPROCESS
# Matcher præcis den pipeline modellen blev trænet med:
# 1) Gråskala 2) Resize til 28x28 3) [0,1]-skalering 4) Evt. invertering
# 5) Normalisering med MNIST-mean/std 6) Batch- og kanal-akse
# -------------------------------------------------------------
def preprocess(img: np.ndarray) -> torch.Tensor:
    # Acceptér både BGR (farve) og gråtone
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Sikr korrekt størrelse (MNIST er 28x28)
    img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)

    # Skaler til [0,1]
    img = img.astype(np.float32) / 255.0

    # MNIST har hvide cifre på sort baggrund. Hvis dit input ligner sort på hvid,
    # så vend farverne (heuristik: gennemsnittet er "lyst")
    if img.mean() > 0.5:
        img = 1.0 - img

    # Til tensor med shape [1,1,28,28] (batch=1, channels=1)
    x = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)

    # Normalisér med MNIST-statistik (samme som i train.py)
    norm = transforms.Normalize((0.1307,), (0.3081,))
    x = norm(x)
    return x


# -------------------------------------------------------------
# 🧠 LOAD MODEL
# Loader vægte fra .pt og sætter modellen i eval-tilstand.
# -------------------------------------------------------------
def load_model(path: str = "mnist_cnn.pt", device: torch.device | None = None):
    device = device or torch.device("cpu")
    model = CNN().to(device)

    # map_location gør det robust, uanset om vægtene blev gemt på GPU/CPU
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, device


# -------------------------------------------------------------
# 🚀 MAIN (CLI)
# Læser billedsti fra argv, preprocesser, kører inferens og printer resultat.
# -------------------------------------------------------------
def main():
    if len(sys.argv) < 2:
        print("Brug: python predict.py PATH_TIL_BILLEDE.png")
        sys.exit(1)

    img_path = sys.argv[1]
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"Kunne ikke læse billedet: {img_path}")
        sys.exit(1)

    model, device = load_model()
    x = preprocess(img).to(device)

    with torch.no_grad():
        logits = model(x)                 # shape: [1, 10]
        probs = torch.softmax(logits, 1)  # gør logits til sandsynligheder
        probs = probs.cpu().numpy()[0]    # til numpy for pæn print

    pred = int(probs.argmax())
    print(f"Predicted: {pred} | probs={np.round(probs, 3)}")


if __name__ == "__main__":
    main()
