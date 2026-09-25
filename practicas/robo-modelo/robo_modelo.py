"""
Model Extraction (Robo de Modelo) - Lab SYP UCM

Fase 1 – Distilación: entrena surrogate imitando al teacher con challenge_bundle.pt
Fase 2 – Extracción: roba el surrogate consultándolo con tiny_train_subset.pt (solo 24 imgs)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split

# ── Dataset ──────────────────────────────────────────────────────────────────

class TeacherDataset(Dataset):
    def __init__(self, images, teacher_probs):
        self.images = images          # [N, 1, 32, 32]
        self.teacher_probs = teacher_probs  # [N, 4] sigmoid outputs

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.teacher_probs[idx]


# ── Modelo Surrogate ──────────────────────────────────────────────────────────

class SurrogateCNN(nn.Module):
    """CNN sencillo que imita al teacher. Salida con sigmoid (igual que el teacher)."""
    def __init__(self, num_classes=4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 16x16
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 8x8
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), # 4x4
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
            nn.Sigmoid(),   # igual que el teacher
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# ── Entrenamiento ─────────────────────────────────────────────────────────────

def train_surrogate(bundle_path="challenge_bundle.pt", epochs=50, batch_size=32, lr=1e-3):
    bundle = torch.load(bundle_path, map_location="cpu", weights_only=False)

    images       = bundle["images"]        # [1200, 1, 32, 32]
    teacher_probs = bundle["teacher_probs"] # [1200, 4]
    label_names  = bundle["label_names"]   # ['circle', 'square', 'triangle', 'cross']

    print(f"Dataset: {len(images)} imágenes, clases: {label_names}")

    dataset = TeacherDataset(images, teacher_probs)

    # 80/20 train/val split (misma seed que el bundle)
    n_train = int(0.8 * len(dataset))
    n_val   = len(dataset) - n_train
    torch.manual_seed(42)
    train_ds, val_ds = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size)

    model = SurrogateCNN(num_classes=len(label_names))
    # BCE porque el teacher usa sigmoid independiente por clase
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for imgs, targets in train_loader:
            preds = model(imgs)
            loss  = criterion(preds, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(imgs)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, targets in val_loader:
                preds    = model(imgs)
                val_loss += criterion(preds, targets).item() * len(imgs)

        train_loss /= n_train
        val_loss   /= n_val
        scheduler.step()

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{epochs}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "surrogate_model.pt")

    print(f"\nMejor val_loss: {best_val_loss:.4f}")
    print("Modelo guardado en surrogate_model.pt")
    return model


# ── Evaluación ─────────────────────────────────────────────────────────────────

def evaluate_agreement(bundle_path="challenge_bundle.pt", model_path="surrogate_model.pt"):
    """Mide qué % de predicciones del surrogate coinciden con las del teacher."""
    bundle = torch.load(bundle_path, map_location="cpu", weights_only=False)
    images        = bundle["images"]
    teacher_probs = bundle["teacher_probs"]
    label_names   = bundle["label_names"]

    model = SurrogateCNN(num_classes=len(label_names))
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    model.eval()

    with torch.no_grad():
        surrogate_probs = model(images)

    # Comparar argmax (clase con mayor probabilidad)
    teacher_pred   = teacher_probs.argmax(dim=1)
    surrogate_pred = surrogate_probs.argmax(dim=1)
    agreement = (teacher_pred == surrogate_pred).float().mean().item()

    print(f"\nAcuerdo surrogate vs teacher: {agreement*100:.1f}%")

    # Distribución de clases predichas por el teacher
    print("\nDistribución clases teacher:")
    for i, name in enumerate(label_names):
        count = (teacher_pred == i).sum().item()
        print(f"  {name}: {count}")

    return agreement


# ── Modelo Extraído ───────────────────────────────────────────────────────────

class ExtractedCNN(nn.Module):
    """Arquitectura del modelo robado — distinta al surrogate para demostrar
    que el atacante no necesita conocer la arquitectura víctima."""
    def __init__(self, num_classes=4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 16x16
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), # 8x8
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), # 4x4
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
            # sin Sigmoid: forward devuelve logits para usar BCEWithLogitsLoss
        )

    def forward(self, x):
        return self.classifier(self.features(x))

    def predict_proba(self, x):
        with torch.no_grad():
            return torch.sigmoid(self.forward(x))


# ── Pérdida de Destilación (igual que el demo de Praetorian) ─────────────────

def distillation_loss(student_logits, teacher_probs, temperature=2.0):
    """BCE con temperature scaling sobre las probabilidades del teacher.
    El factor T^2 compensa el escalado del gradiente (Hinton et al., 2015)."""
    clipped  = teacher_probs.clamp(1e-4, 1 - 1e-4)
    softened = torch.sigmoid(torch.logit(clipped) / temperature)  # soft targets
    return F.binary_cross_entropy_with_logits(
        student_logits / temperature, softened
    ) * (temperature ** 2)


# ── Fase 2: Ataque de Extracción ──────────────────────────────────────────────

def extract_model(
    victim_path="surrogate_model.pt",
    tiny_bundle_path="tiny_train_subset.pt",
    eval_bundle_path="challenge_bundle.pt",
    output_path="extracted_model.pt",
    epochs=150,
    batch_size=8,
    lr=1e-3,
    temperature=2.0,
):
    """
    Roba el modelo víctima usando solo tiny_train_subset como consultas black-box.

    Flujo:
      1. Cargar víctima (surrogate_model.pt) — caja negra
      2. Consultar con las 24 imágenes de tiny_train_subset → soft labels
      3. Entrenar ExtractedCNN con distilación sobre esas 24 consultas
      4. Evaluar acuerdo en los 1200 imágenes del challenge_bundle
    """
    # 1. Cargar víctima
    victim = SurrogateCNN(num_classes=4)
    victim.load_state_dict(torch.load(victim_path, map_location="cpu", weights_only=True))
    victim.eval()

    # 2. Consultar la víctima (simula llamadas a la API black-box)
    tiny = torch.load(tiny_bundle_path, map_location="cpu", weights_only=False)
    query_images = tiny["images"]                   # [24, 1, 32, 32]
    label_names  = tiny["label_names"]

    with torch.no_grad():
        query_probs = victim(query_images)          # [24, 4] — probabilidades del victim

    print(f"Consultas al victim: {len(query_images)} imágenes")
    print(f"Clases: {label_names}")

    # 3. Entrenar el modelo extraído con esas consultas
    dataset = TensorDataset(query_images, query_probs)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    extracted = ExtractedCNN(num_classes=len(label_names))
    optimizer = optim.Adam(extracted.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(1, epochs + 1):
        extracted.train()
        epoch_loss = 0.0
        for imgs, probs in loader:
            logits = extracted(imgs)
            loss   = distillation_loss(logits, probs, temperature)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(imgs)

        scheduler.step()
        if epoch % 30 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{epochs}  loss={epoch_loss/len(query_images):.4f}")

    torch.save(extracted.state_dict(), output_path)
    print(f"\nModelo extraído guardado en {output_path}")

    # 4. Evaluar en challenge_bundle (1200 imágenes)
    eval_bundle = torch.load(eval_bundle_path, map_location="cpu", weights_only=False)
    eval_images       = eval_bundle["images"]
    eval_teacher_probs = eval_bundle["teacher_probs"]

    extracted.eval()
    with torch.no_grad():
        extracted_probs = extracted.predict_proba(eval_images)   # [1200, 4]

    # Acuerdo exacto (argmax)
    victim_pred    = eval_teacher_probs.argmax(dim=1)
    extracted_pred = extracted_probs.argmax(dim=1)
    exact_agreement = (victim_pred == extracted_pred).float().mean().item()

    # MAE de probabilidades
    prob_mae = (extracted_probs - eval_teacher_probs).abs().mean().item()

    print("\n=== Resultados extracción ===")
    print(f"Consultas usadas:        {len(query_images)}")
    print(f"Imágenes de evaluación:  {len(eval_images)}")
    print(f"Acuerdo exacto:          {exact_agreement*100:.1f}%")
    print(f"MAE probabilidades:      {prob_mae:.4f}")
    print("=" * 32)

    return extracted


if __name__ == "__main__":
    # Fase 1: entrenar surrogate por distilación del teacher
    print("=" * 48)
    print("FASE 1 — Distilación del teacher")
    print("=" * 48)
    train_surrogate(epochs=60)
    evaluate_agreement()

    # Fase 2: robar el surrogate con solo 24 consultas
    print("\n" + "=" * 48)
    print("FASE 2 — Extracción con tiny_train_subset")
    print("=" * 48)
    extract_model()
