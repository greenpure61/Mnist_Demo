MNIST CV Demo — Håndskrevet tal (0–9) med PyTorch + OpenCV

Et lille, portfolio-venligt computer vision-projekt der viser hele kæden: data → model → evaluering → demo.
Du kan træne en CNN på MNIST, forudsige fra filer via CLI, og teste live i en OpenCV-tegneplade (skriv et tal med musen → få et gæt).

🚀 Features

Simpel, robust CNN (Conv+BN+ReLU+Pool ×2 → MLP+Dropout)

Data augmentation (rotation/shift/scale/sheer) for bedre generalisering

OpenCV demo med tegneplade, eraser, top-3 overlay og preview af 28×28-input

MNIST-lignende preprocess: crop → bevar aspect ratio → pad → center efter massecentrum

CLI-predict for enkeltbilleder

Gemmer bedste model automatisk (mnist_cnn.pt)

🗂️ Projektstruktur
.
├─ data/                 # MNIST downloades her automatisk
├─ train.py              # træning + evaluering + gem model
├─ predict.py            # CLI: forudsig fra et billede
└─ draw_pad.py           # OpenCV-tegneplade med overlay & preview
