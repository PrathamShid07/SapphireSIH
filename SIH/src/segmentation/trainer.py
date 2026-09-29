from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def train_segmentation(model, train_dataset, output_dir, config_manager=None, logger=None):
    output_dir = Path(output_dir)
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    final_ckpt = ckpt_dir / "model_final.pth"

    if config_manager:
        epochs = config_manager.get("segmentation.epochs", 5)
        batch_size = config_manager.get("segmentation.batch_size", 2)
        lr = config_manager.get("segmentation.learning_rate", 0.0001)
        resume = config_manager.get("execution.resume", True)
    else:
        epochs, batch_size, lr, resume = 5, 2, 0.0001, True

    if resume and final_ckpt.exists():
        if logger:
            logger.info(f"Checkpoint found at {final_ckpt}. Skipping training (resume=True).")
        model.load(final_ckpt)
        return {"skipped": True, "checkpoint": str(final_ckpt)}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        for imgs, labels, _ in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out["out"], labels)
            if model._model.aux_classifier is not None and "aux" in out:
                loss += 0.4 * criterion(out["aux"], labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / max(len(loader), 1)
        history.append({"epoch": epoch, "loss": avg_loss})
        if logger:
            logger.info(f"Epoch {epoch}/{epochs} — loss: {avg_loss:.4f}")

    model.save(final_ckpt)
    if logger:
        logger.info(f"Checkpoint saved: {final_ckpt}")

    return {"skipped": False, "checkpoint": str(final_ckpt), "epochs": epochs, "history": history}
