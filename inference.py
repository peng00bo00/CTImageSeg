import argparse
import torch.nn.functional
import yaml
import os
import cv2
import numpy as np
from tqdm import tqdm

import torch

from seg import create_predefined_model, create_smp_predefined_model, pad_to_divisible_by_32


def inference(yml_path, pth_path, img_path, save_path):
    ## parse .yml file to retrieve dataset/model/training parameters
    with open(yml_path, "r") as f:
        yml = yaml.safe_load(f)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_classes = len(yml["DataSet"]["labels"])

    print("Creating output dirs...")
    for idx in range(num_classes):
        label_save_path = os.path.join(save_path, f"label_{idx}")
        os.makedirs(label_save_path, exist_ok=True)

    ## create model instance
    print("Initializing seg model...")
    model_params = yml["Model"]

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

    ## load trained model
    model = model.to(device)
    model.load_state_dict(torch.load(pth_path))
    model.eval()

    print("Seg model is loaded!")

    ## inference on CT images
    imgs = [img for img in os.listdir(img_path) if img.endswith((".jpg", ".png", ".bmp"))]

    for name in tqdm(imgs):
        img = cv2.imread(os.path.join(img_path, name), 0)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        ## image to tensor
        img = torch.from_numpy(img[np.newaxis, :, :, :])

        ## from [N, H, W, C] to [N, C, H, W] layout
        img = img.permute([0, 3, 1, 2]).float()
        img = pad_to_divisible_by_32(img)

        with torch.no_grad():
            img  = img.to(device)
            pred = model(img)
            pred = torch.sigmoid(pred)

            ## detach and transform to numpy
            pred = pred.detach().squeeze().permute([1, 2, 0])
            pred = pred.cpu().numpy()
        
        ## convert prediction percentage confidence
        pred = pred * 100

        for idx in range(num_classes):
            save_file_path = os.path.join(save_path, f"label_{idx}", f"{name}.png")
            cv2.imwrite(save_file_path, pred[:, :, idx].astype(np.uint8))
        
        # tqdm.write(f"{name} is saved in {save_path}")

def main():
    parser = argparse.ArgumentParser(description="A script to start training process.")
    parser.add_argument('--yml', type=str, help='path to .yml file')
    parser.add_argument('--pth', type=str, help='path to .pth file')
    parser.add_argument('--img', type=str, help='path to CT images')
    parser.add_argument('--save_path', type=str, help='save path')

    args = parser.parse_args()
    yml_path = args.yml
    pth_path = args.pth
    img_path = args.img
    save_path= args.save_path

    inference(yml_path, pth_path, img_path, save_path)

if __name__ == "__main__":
    main()