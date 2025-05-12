import yaml
import argparse

import torch
import wandb
import kagglehub

from dataset import get_diatom_dataset, get_diatom_dataloader, classes
from model import get_fasterrcnn_model, get_retinanet_model
from trainer import Trainer

device = torch.device(
  "cuda" if torch.cuda.is_available() else
  "mps" if torch.backends.mps.is_available() else
  "cpu"
)

def main():
  parser = argparse.ArgumentParser(description="Training configuration for object detection")

  parser.add_argument(
      "--checkpoint", type=str, required=False, default=None,
      help="Directory to save model checkpoints"
  )

  parser.add_argument(
      "--config", type=str, required=True,
      help="Configuratino .yaml file for Weights & Biases"
  )

  parser.add_argument(
    "--epoch", type=int, required=True,
    help="Number of training epochs"
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
  
  if args.checkpoint is None:
    args.checkpoint = f"{config['model']}-{group_name}"

  try:
    trainer = Trainer.from_checkpoint(
      model,
      optimizer,
      train_dataloader,
      val_dataloader,
      args.checkpoint,
      device
    )

    print(f"Resumed training from checkpoint in: {args.checkpoint}")

  except Exception as e:
    print(f"Failed to load checkpoint: {e}")
    print(f"Starting new training from scratch in: {args.checkpoint}")

    trainer = Trainer(
      model,
      optimizer,
      train_dataloader,
      val_dataloader,
      args.checkpoint,
      device
    )

  wandb.init(
    project=config["model"],
    group=group_name,
    config=config,
  )

  trainer.train(args.epoch)

  wandb.finish()

if __name__ == "__main__":
  main()
