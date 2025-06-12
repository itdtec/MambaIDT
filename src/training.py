from .model import MambaITD
from .utils_ import otsu_threshold
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, accuracy_score
from .config import load_config
import os

cfg_set = load_config()["experiment"]
cfg_path = load_config()["paths"]


def validate(model, val_loader):
    model.eval()
    all_scores, all_labels = [], []
    with torch.no_grad():
        for S_b, S_c, X, y in val_loader:
            logits = model(S_b, S_c, X)
            scores = torch.sigmoid(logits).mean(dim=1).cpu().numpy()
            all_scores.extend(scores)
            all_labels.extend(y.cpu().numpy())
    threshold = otsu_threshold(all_scores)
    preds = [1 if s >= threshold else 0 for s in all_scores]
    tn, fp, fn, tp = confusion_matrix(all_labels, preds).ravel()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0   # detection rate
    f1        = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    fpr       = fp / (fp + tn) if (fp + tn) > 0 else 0.0  # false positive rate
    acc       = accuracy_score(all_labels, preds)

    print(
        f"[Val-OTSU] Thresh={threshold:.4f}  "
        f"P={precision:.4f}  "
        f"DR={recall:.4f}  "
        f"F1={f1:.4f}  "
        f"FPR={fpr:.4f}  "
        f"Acc={acc:.4f}"
    )

def train_model(train_dataset, val_dataset):
    model = MambaITD()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg_set["learning_rate"])
    criterion = torch.nn.BCEWithLogitsLoss()
    train_loader = DataLoader(train_dataset, batch_size=cfg_set["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg_set["batch_size"], shuffle=False)

    for epoch in range(cfg_set["epochs"]):
        model.train()
        total_loss = 0

        for S_b, S_c, X, y in train_loader:
            optimizer.zero_grad()
            logits = model(S_b, S_c, X)
            y_exp = y.unsqueeze(1).expand(-1, logits.size(1))
            loss_main = criterion(logits, y_exp)
            loss_gate = model.get_gate_regularization_loss()
            loss = loss_main + cfg_set["lambda_gate"] * loss_gate
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"[Epoch {epoch+1}] Loss: {avg_loss:.6f}")
        validate(model, val_loader)

        # Early stopping condition
        if avg_loss < 0.01:
            print(f"Early stopping: loss={avg_loss:.6f} < 0.0005")
            save_dir = "checkpoints"
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, "mamba_itd_best.pth")
            torch.save(model.state_dict(), save_path)
            break

    return model

