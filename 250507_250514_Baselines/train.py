import torch
import wandb
import kagglehub

from dataset import get_diatom_dataset, get_diatom_dataloader, classes
from model import get_fasterrcnn_model
from trainer import Trainer

def main():
  diatom_dataset_root_dir = kagglehub.dataset_download("huseyingunduz/diatom-dataset")

  train_dataset, val_dataset = get_diatom_dataset(root_dir=diatom_dataset_root_dir)
  train_dataloader, val_dataloader = get_diatom_dataloader(train_dataset, val_dataset)

  device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
  )

  wandb.init(project="your_project_name")

  model = get_fasterrcnn_model(len(classes) + 1)
  model.to(device)

  for param in model.backbone.parameters():
      param.requires_grad = False

  optimizer = torch.optim.SGD(
    [p for p in model.parameters() if p.requires_grad],
    lr=0.005,
    momentum=0.9,
  )

  trainer = Trainer(
    model,
    optimizer,
    train_dataloader,
    val_dataloader,
    "/content/checkpoints",
    device
  )
  
  trainer.train(5)

if __name__ == "__main__":
  main()
