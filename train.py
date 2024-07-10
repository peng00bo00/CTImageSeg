import argparse
import yaml

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split, DataLoader
from torchvision import transforms

from seg import CTImageDataset, create_predefined_model, SegModelTrainer, FocalLoss


def train(yml_path):
    ## parse .yml file to retrieve dataset/model/training parameters
    with open("./yml/fcn_predefined.yml", "r") as f:
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

    transform= None
    if "transform" in dataset_params:
        transform_list = []

        if dataset_params["transform"].get("crop", False):
            crop_size = dataset_params["transform"]
            transform_list.append(transforms.RandomCrop(crop_size))
        
        if dataset_params["transform"].get("flip", False):
            transform_list.append(transforms.RandomHorizontalFlip())
            transform_list.append(transforms.RandomVerticalFlip())

        transform = transforms.Compose(transform_list)

    train_dataset = CTImageDataset(img_root, xml_paths, labels, thickness, transform)
    val_dataset   = CTImageDataset(img_root, xml_paths, labels, thickness, None)

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

    ## dataloader
    batch_size = dataset_params.get("batch_size", 16)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

    ## create model
    if model_params.get("predefined", False):
        model = create_predefined_model(model_params["name"], num_classes=num_classes)
    ## TODO: update with customized model later
    else:
        pass
    
    lr = model_params.get("lr", 1e-3)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    ## create trainer
    if trainer_params.get("loss_fn", "sigmoid") == "focal":
        trainer_params["loss_fn"] = FocalLoss()
    else:
        trainer_params["loss_fn"] = nn.BCEWithLogitsLoss()
    
    trainer = SegModelTrainer(trainer_params)

    ## start training
    trainer.fit(model, train_loader, val_loader, optimizer)

def main():
    parser = argparse.ArgumentParser(description="A script to start training process.")
    parser.add_argument('--yml', type=str, help='path to .yml file')

    args = parser.parse_args()
    yml_path = args.yml

    train(yml_path)

if __name__ == "__main__":
    main()