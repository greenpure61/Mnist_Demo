"""
draw_pad.py
-----------
En simpel OpenCV-tegneplade til MNIST-demoen.
Kontroller:
    - Venstre musetast: tegn
    - 'p' : predict (skriv resultat i terminalen)
    - 'c' : clear (nulstil canvas)
    - 'q' : quit (luk programmet)
"""

import cv2
import numpy as np
import torch

from predict import load_model, preprocess  # genbrug samme model og preprocess


# Canvasstørrelse (større = nemmere at tegne; vi resizer ned til 28x28 ved predict)
W, H = 280, 280
BRUSH_RADIUS = 8

# Global state for mouse callback
drawing = False
last_pt: tuple[int, int] | None = None
canvas = None  # init senere


# -------------------------------------------------------------
# 🖱️ MOUSE CALLBACK
# Tegner tykke linjer i hvid (255) på sort baggrund (0).
# Vi gemmer sidste punkt for at lave "glatte" linjer under musebevægelse.
# -------------------------------------------------------------
def on_mouse(event, x, y, flags, param):
    global drawing, last_pt, canvas

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        last_pt = (x, y)
        cv2.circle(canvas, (x, y), BRUSH_RADIUS, 255, -1)

    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        cv2.line(canvas, last_pt, (x, y), 255, thickness=BRUSH_RADIUS * 2)
        last_pt = (x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        last_pt = None


# -------------------------------------------------------------
# 🔮 PREDICT CURRENT CANVAS
# Tager nuværende canvas, resizer til 28x28, preprocesser og kører inferens.
# Printer både prediction og klasse-sandsynligheder.
# -------------------------------------------------------------
@torch.no_grad()
def predict_canvas(model: torch.nn.Module, device: torch.device):
    #lav en kopi så vi ikke ændre selve canvas
    img = canvas.copy()

    #tilføjer Gaussian blur for at blødgøre kanter
    img = cv2.GaussianBlur(img, (3, 3), 0)

    # automatisk kontrastjustering
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)

    # Resize til MNIST-størrelse
    small = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)

    # Brug samme preprocess som i predict.py (inkl. evt. inversion + normalisering)
    x = preprocess(small).to(device)

    logits = model(x)
    probs = torch.softmax(logits, 1).cpu().numpy()[0]
    pred = int(probs.argmax())

    print(f"[Predict] -> {pred}  probs: {np.round(probs, 3)}")


# -------------------------------------------------------------
# 🚀 MAIN LOOP
# Opretter vindue og model, og lytter efter keyboard-events.
# -------------------------------------------------------------
def main():
    global canvas
    canvas = np.zeros((H, W), dtype=np.uint8)  # sort baggrund

    cv2.namedWindow("MNIST Draw")
    cv2.setMouseCallback("MNIST Draw", on_mouse)

    # Load model én gang (hurtigere end at loade pr. predict)
    model, device = load_model()

    while True:
        # Farvevisning (BGR) så vi kan lægge tekst i lysere tone
        view = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
        cv2.putText(view, "Draw a digit.  p:predict  c:clear  q:quit",
                    (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)
        cv2.imshow("MNIST Draw", view)

        key = cv2.waitKey(10) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            canvas[:] = 0
        elif key == ord('p'):
            predict_canvas(model, device)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
