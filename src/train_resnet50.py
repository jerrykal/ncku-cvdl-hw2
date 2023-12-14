import os
from argparse import ArgumentParser, Namespace

import torch
import torchvision
from matplotlib import pyplot as plt
from torchvision.transforms import v2

from cat_dog_dataset import CatDogDataset
from custom_resnet50 import CustomResNet50

transform = v2.Compose(
    [
        v2.Resize(244),
        v2.CenterCrop(224),
        v2.RandomHorizontalFlip(),
        v2.RandomVerticalFlip(),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
    ]
)

transform_re = v2.Compose(
    [
        v2.Resize(244),
        v2.CenterCrop(224),
        v2.RandomHorizontalFlip(),
        v2.RandomVerticalFlip(),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.RandomErasing(),
    ]
)

transform_val = v2.Compose(
    [
        v2.Resize(244),
        v2.CenterCrop(224),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
    ]
)


def parse_args() -> Namespace:
    parser = ArgumentParser()

    # Paths
    parser.add_argument(
        "--data_dir",
        type=str,
        # ../data
        default=os.path.abspath(os.path.join(__file__, os.pardir, os.pardir, "data")),
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        # ../models
        default=os.path.abspath(os.path.join(__file__, os.pardir, os.pardir, "models")),
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        # ../logs
        default=os.path.abspath(os.path.join(__file__, os.pardir, os.pardir, "logs")),
    )

    # Model parameters
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--random_erasing", action="store_true")

    return parser.parse_args()


def main(args: Namespace) -> None:
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    # Load dataset
    train_dataset = CatDogDataset(
        root_dir=os.path.abspath(
            os.path.join(
                __file__, os.pardir, os.pardir, "data", "Q5", "training_dataset"
            )
        ),
        transform=transform if not args.random_erasing else transform_re,
    )
    val_dataset = CatDogDataset(
        root_dir=os.path.abspath(
            os.path.join(
                __file__, os.pardir, os.pardir, "data", "Q5", "validation_dataset"
            )
        ),
        transform=transform_val,
    )

    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False
    )

    # Create resnet50 model for binary classification
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CustomResNet50().to(device)

    # Create loss function and optimizer
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr)

    # Start training
    best_val_acc = None
    for epoch in range(args.num_epochs):
        epoch_train_loss = 0.0
        epoch_train_acc = 0.0

        # Train
        model.train()
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            # The output shape would be (batch_size, 1), thus we need to squeeze it
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_train_acc += (outputs.round() == labels).sum().item()
            epoch_train_loss += loss.item()

        epoch_train_acc /= len(train_dataset)
        epoch_train_loss /= len(train_loader)

        # Validate
        with torch.no_grad():
            epoch_val_loss = 0.0
            epoch_val_acc = 0.0

            model.eval()
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs).squeeze()
                loss = criterion(outputs, labels)

                epoch_val_acc += (outputs.round() == labels).sum().item()
                epoch_val_loss += loss.item()

            epoch_val_acc /= len(val_dataset)
            epoch_val_loss /= len(val_loader)

            # Save model
            if best_val_acc is None or epoch_val_acc > best_val_acc:
                best_val_acc = epoch_val_acc
                torch.save(
                    model.state_dict(),
                    os.path.join(
                        args.save_dir,
                        "resnet50.pth"
                        if not args.random_erasing
                        else "resnet50_re.pth",
                    ),
                )
                print(f"Saved model with validation accuracy: {best_val_acc:.4f}")

        print(
            f"Epoch {epoch + 1}/{args.num_epochs} | "
            f"Train Loss: {epoch_train_loss:.4f} | "
            f"Train Acc: {epoch_train_acc:.4f} | "
            f"Val Loss: {epoch_val_loss:.4f} | "
            f"Val Acc: {epoch_val_acc:.4f}"
        )


if __name__ == "__main__":
    main(parse_args())
