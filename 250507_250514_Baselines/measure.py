import os
import json
import yaml
import argparse

import torch
import wandb
import kagglehub

from dataset import get_diatom_dataset, get_diatom_dataloader, classes
from model import get_fasterrcnn_model, get_retinanet_model
from trainer import Trainer, validate_epoch, to_device

from tqdm.auto import tqdm

device = torch.device(
  "cuda" if torch.cuda.is_available() else
  "mps" if torch.backends.mps.is_available() else
  "cpu"
)

@torch.no_grad()
def loss_epoch(model, dataloader, device):
  total_loss = 0

  model.train()

  for images, targets in tqdm(dataloader, desc="Training"):
    images, targets = to_device(images, targets, device)

    loss_dict = model(images, targets)
    losses = sum(loss for loss in loss_dict.values())

    total_loss += losses.item()

  return total_loss / len(dataloader)

def main():
  parser = argparse.ArgumentParser(description="")

  parser.add_argument(
      "--config", type=str, required=True,
      help="Configuratino .yaml file for Weights & Biases"
  )

  parser.add_argument(
    "--batch", type=int, required=False, default=4,
    help="Batch size"
  )

  parser.add_argument(
    "--worker", type=int, required=False, default=6,
    help="Number of data loader workers"
  )

  parser.add_argument(
    "--prefetch", type=int, required=False, default=4,
    help="Prefetch factor for DataLoader"
  )

  args = parser.parse_args()

  with open(args.config, "r") as f:
    config = yaml.safe_load(f)

  diatom_dataset_root_dir = kagglehub.dataset_download("huseyingunduz/diatom-dataset")

  train_dataset, val_dataset = get_diatom_dataset(root_dir=diatom_dataset_root_dir)
  train_dataloader, val_dataloader = get_diatom_dataloader(train_dataset, val_dataset, args.batch, args.worker, args.prefetch)

  if config["model"] == "fasterrcnn":
    model = get_fasterrcnn_model(len(classes) + 1)
  elif config["model"] == "retinanet":
    model = get_retinanet_model(len(classes) + 1)

  model.to(device)

  if config["mode"] == "full":
    optimizer = torch.optim.SGD(
      [p for p in model.parameters() if p.requires_grad],
      lr=config["learning_rate"],
      momentum=0.9,
    )

  elif config["mode"] == "head":
    for param in model.backbone.parameters():
      param.requires_grad = False

    optimizer = torch.optim.SGD(
      [p for p in model.parameters() if p.requires_grad],
      lr=config["learning_rate"],
      momentum=0.9,
    )

  elif config["mode"] == "split":
    backbone_params = []
    head_params = []

    for name, param in model.named_parameters():
      if not param.requires_grad:
        continue
      if name.startswith("backbone"):
        backbone_params.append(param)
      elif name.startswith("head"):
        head_params.append(param)

    optimizer = torch.optim.SGD(
      [
        {"params": backbone_params, "lr": config["back_lr"]},
        {"params": head_params, "lr": config["head_lr"]},
      ],
      momentum=0.9
    )

  group_name = f"{config['mode']}-b{config['back_lr']:.0e}-h{config['head_lr']:.0e}" \
    if config["mode"] == "split" else config["mode"]
  
  args.checkpoint = f"{config['model']}-{group_name}"
  
  checkpoints_path = os.path.join(args.checkpoint, "checkpoints.json")
  with open(checkpoints_path, "r") as f:
    checkpoints = {int(k): v for k, v in json.load(f).items()}

  sorted_keys = sorted(checkpoints.keys())

  wandb.init(
    project=config["model"],
    group=group_name,
    config=config,
  )

  val_losses = []
  train_scores = []

  for epoch in sorted_keys:
    checkpoint_path = checkpoints[epoch]
    checkpoint = torch.load(checkpoint_path)

    model.load_state_dict(checkpoint["model_state_dict"])

    val_loss = loss_epoch(model, val_dataloader, device)
    train_score = validate_epoch(model, train_dataloader, device)

    val_losses.append(val_loss)
    train_scores.append(train_score)
    
    wandb.log({
      "epoch": epoch,
      "val/loss": val_loss,
      "train/map": train_score["map"].item(),
      "train/map_50": train_score["map_50"].item(),
      "train/map_75": train_score["map_75"].item(),
    })

    print(f"Epoch: {epoch} | "
      f"Val Loss: {val_loss:.3f} | "
      f"Train mAP: {train_score['map'].item():.3f} | "
      f"Train mAP@50: {train_score['map_50'].item():.3f} | "
      f"Train mAP@75: {train_score['map_75'].item():.3f}"
    )

  torch.save(
    {
      "val_losses": val_losses,
      "train_scores": train_scores
    },
    os.path.join("measures", f"{config['model']}-{group_name}-measure.pt")
  )

  wandb.finish()

if __name__ == "__main__":
  main()
