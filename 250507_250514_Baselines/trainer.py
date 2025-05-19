import os
import json
import uuid

import wandb

import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from tqdm.auto import tqdm

def to_device(images, targets, device):
  return [image.to(device) for image in images], [{k: v.to(device) for k, v in target.items()} for target in targets]

def train_epoch(model, optimizer, dataloader, device):
  total_loss = 0

  model.train()

  for images, targets in tqdm(dataloader, desc="Training"):
    images, targets = to_device(images, targets, device)

    loss_dict = model(images, targets)
    losses = sum(loss for loss in loss_dict.values())

    optimizer.zero_grad()
    losses.backward()
    optimizer.step()

    total_loss += losses.item()

  return total_loss / len(dataloader)

def validate_epoch(model, dataloader, device):
  mean_ap = MeanAveragePrecision().to(device)

  model.eval()

  with torch.no_grad():
    for images, targets in tqdm(dataloader, desc="Validating"):
      images, targets = to_device(images, targets, device)

      outputs = model(images)

      mean_ap.update(outputs, targets)

  score = mean_ap.compute()
  score_cpu = {k: v.cpu() if torch.is_tensor(v) else v for k, v in score.items()}

  return score_cpu

class Trainer:
  def __init__(self, model, optimizer, train_dataloader, val_dataloader, root_dir, device):
    self.model = model
    self.optimizer = optimizer
    self.train_dataloader = train_dataloader
    self.val_dataloader = val_dataloader
    self.root_dir = root_dir
    self.device = device

    self.complete_epoch = 0

    self.train_losses = []
    self.scores = []

    self.checkpoints = {}

  @staticmethod
  def from_checkpoint(model, optimizer, train_dataloader, val_dataloader, root_dir, device):
    os.makedirs(root_dir, exist_ok=True)
    
    if not os.path.isfile(os.path.join(root_dir, "checkpoints.json")):
      raise FileNotFoundError(f"No checkpoints.json found in {root_dir}")

    with open(os.path.join(root_dir, "checkpoints.json"), "r") as f:
      checkpoints = {int(k): v for k, v in json.load(f).items()}

    checkpoint_path = checkpoints[max(checkpoints.keys())]
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    trainer = Trainer(model, optimizer, train_dataloader, val_dataloader, root_dir, device)

    trainer.complete_epoch = checkpoint["epoch"]
    trainer.train_losses = checkpoint["train_losses"]
    trainer.scores = checkpoint["scores"]

    trainer.checkpoints = checkpoints

    return trainer

  def train(self, epochs):
    start_epoch = self.complete_epoch + 1

    for epoch in range(start_epoch, start_epoch + epochs):
      loss = train_epoch(self.model, self.optimizer, self.train_dataloader, self.device)
      score = validate_epoch(self.model, self.val_dataloader, self.device)

      self.train_losses.append(loss)
      self.scores.append(score)

      self.complete_epoch = epoch

      self.checkout()

      wandb.log({
        "epoch": epoch,
        "train/loss": loss,
        "val/map": score["map"].item(),
        "val/map_50": score["map_50"].item(),
        "val/map_75": score["map_75"].item(),
      })

      print(f"Epoch: {epoch} | "
        f"Train Loss: {loss:.3f} | "
        f"Val mAP: {score['map'].item():.3f} | "
        f"Val mAP@50: {score['map_50'].item():.3f} | "
        f"Val mAP@75: {score['map_75'].item():.3f}"
      )

  def checkout(self):
    os.makedirs(self.root_dir, exist_ok=True)

    output_path = os.path.join(self.root_dir, f"epoch{self.complete_epoch}.pt")

    torch.save(
      {
        "model_state_dict": self.model.state_dict(),
        "optimizer_state_dict": self.optimizer.state_dict(),
        "epoch": self.complete_epoch,
        "train_losses": self.train_losses,
        "scores": self.scores,
      },
      output_path
    )

    self.checkpoints[self.complete_epoch] = output_path

    with open(os.path.join(self.root_dir, "checkpoints.json"), "w") as f:
      f.write(json.dumps(self.checkpoints))

  def evaluate(self):
    score = validate_epoch(self.model, self.val_dataloader)

    print(f"Val mAP: {score['map'].item():.3f} | "
      f"Val mAP@50: {score['map_50'].item():.3f} | "
      f"Val mAP@75: {score['map_75'].item():.3f}"
    )

    return score
  