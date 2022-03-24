import torch
from torch.utils.data import DataLoader

from expert_dataset import ExpertDataset
from models.cilrs import CILRS


def validate(model, dataloader):
    """Validate model performance on the validation dataset"""
    # Your code here
    pass


def train(model, dataloader):
    """Train model on the training dataset for one epoch"""
    # Your code here
    pass


def plot_losses(train_loss, val_loss):
    """Visualize your plots and save them for your report."""
    # Your code here
    pass


def main():
    # Change these paths to the correct paths in your downloaded expert dataset
    train_root = None
    val_root = None
    model = CILRS()
    train_dataset = ExpertDataset(train_root)
    val_dataset = ExpertDataset(val_root)

    # You can change these hyper parameters freely, and you can add more
    num_epochs = 10
    batch_size = 64
    save_path = "cilrs_model.ckpt"

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    train_losses = []
    val_losses = []
    for i in range(num_epochs):
        train_losses.append(train(model, train_loader))
        val_losses.append(validate(model, val_loader))
    torch.save(model, save_path)
    plot_losses(train_losses, val_losses)


if __name__ == "__main__":
    main()
