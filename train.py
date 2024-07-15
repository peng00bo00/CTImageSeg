import argparse
import yaml
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split, DataLoader
from torchvision import transforms

from seg import CTImageDataset, SegModelTrainer, FocalLoss
from seg import create_predefined_model, create_smp_predefined_model, create_lr_scheduler, create_loss_fn
from seg import pad_to_divisible_by_32

def train(yml_path):
    ## parse .yml file to retrieve dataset/model/training parameters
    with open(yml_path, "r") as f:
        yml = yaml.safe_load(f)

    dataset_params = yml["DataSet"]
    model_params   = yml["Model"]
    trainer_params = yml["Trainer"]

    ## create dataset
    img_root = dataset_params["img_root"]
    xml_paths= dataset_params["xml_paths"]
    labels   = dataset_params["labels"]
    thickness= dataset_params.get("thickness", 2)
    num_classes = len(labels)
    xml_paths = [os.path.join(xml_paths, xml_file_path) for xml_file_path in os.listdir(xml_paths) if xml_file_path.endswith(".xml")]

    transform= None
    if "transform" in dataset_params:
        transform_list = []

        if dataset_params["transform"].get("crop", False):
            crop_size = dataset_params["transform"]["crop"]
            transform_list.append(transforms.RandomCrop(crop_size))
        
        if dataset_params["transform"].get("flip", False):
            transform_list.append(transforms.RandomHorizontalFlip())
            transform_list.append(transforms.RandomVerticalFlip())

        transform = transforms.Compose(transform_list)

    train_dataset = CTImageDataset(img_root, xml_paths, labels, thickness, transform)
    val_dataset   = CTImageDataset(img_root, xml_paths, labels, thickness, transforms.Lambda(pad_to_divisible_by_32))

    ## split training set and validation set
    train_ratio = 0.8
    train_size = int(train_ratio * len(train_dataset))
    val_size = len(train_dataset) - train_size

    ## set random seed manually
    rng_state = torch.get_rng_state()
    torch.manual_seed(20240710)

    train_dataset, _ = random_split(train_dataset, [train_size, val_size])
    _, val_dataset = random_split(val_dataset, [train_size, val_size])

    torch.set_rng_state(rng_state)

    print(f"Train DataSet Size: {len(train_dataset)}")
    print(f"Val DataSet Size: {len(val_dataset)}")

    ## dataloader
    batch_size = dataset_params.get("batch_size", 16)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader   = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=8, pin_memory=True)
    print("Dataloader preparation is complete!")

    ## create model
    if model_params.get("predefined", False):
        if model_params["predefined"] == "smp":
            model = create_smp_predefined_model(model_params["model_type"], num_classes=num_classes, parameters=model_params["parameters"])
            model_type = model_params["model_type"]
            print(f"Using SMP.{model_type}() as segmentation model.")
        else:
            model = create_predefined_model(model_params["model_name"], num_classes=num_classes)
    ## TODO: update with customized model later
    else:
        pass
    
    model.to(trainer_params["device"])
    
    ## create trainer
    ## loss function
    loss_params = trainer_params.get("loss", {})
    if loss_params.get("predefined", False) == "smp":
        trainer_params["loss_fn"] = create_loss_fn(loss_params["loss_type"], loss_params.get("parameters", {}))

        loss_type = loss_params["loss_type"]
        print(f"Using SMP.{loss_type}() for training.")

    # reduction = trainer_params.get("reduction", "mean")
    # if trainer_params.get("loss_fn", "sigmoid") == "focal":
    #     trainer_params["loss_fn"] = FocalLoss(reduction=reduction)
    #     print(f"Using FocalLoss for training (reduction={reduction}).")
    # else:
    #     trainer_params["loss_fn"] = nn.BCEWithLogitsLoss(reduction=reduction)
    #     print(f"Using BCEWithLogitsLoss for training (reduction={reduction}).")
    
    ## optimizer
    lr = trainer_params.get("lr", 1e-3)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    ## scheduler
    scheduler = create_lr_scheduler(optimizer, trainer_params.get("scheduler", {}))
    
    ## trainer instance
    trainer_params["model_name"] = model_params["model_name"]
    trainer_params["num_classes"] = num_classes

    trainer = SegModelTrainer(trainer_params)

    ## start training
    trainer.fit(model, train_loader, val_loader, optimizer, scheduler)
    # trainer._evaluate(model, val_loader, 0)

def main():
    parser = argparse.ArgumentParser(description="A script to start training process.")
    parser.add_argument('--yml', type=str, help='path to .yml file')

    args = parser.parse_args()
    yml_path = args.yml

    train(yml_path)

if __name__ == "__main__":
    main()