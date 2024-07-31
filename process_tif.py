import os
import cv2
import yaml
import torch
import numpy as np
from tqdm import tqdm

from seg import create_predefined_model, create_smp_predefined_model, pad_to_divisible_by_32


def recursive_slect_tif(root, paths):
    """Slect the tif images from root recursively.

    Args:
        root: current root
        paths: selected tif image path

    Return:
        paths: tif image path
    """

    for file in os.listdir(root):
        path = os.path.join(root, file)

        if file.endswith(".tif"):
            paths.append(path)
        elif os.path.isdir(path):
            paths = recursive_slect_tif(path, paths)

    return paths

def process_tif_frames(frames, model, num_classes):
    seg_frames = [[] for _ in range(num_classes)]
    for img in tqdm(frames):
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        img = torch.from_numpy(img[np.newaxis, :, :, :])

        ## from [N, H, W, C] to [N, C, H, W] layout
        img = img.permute([0, 3, 1, 2]).float()
        img = pad_to_divisible_by_32(img)

        with torch.no_grad():
            img  = img.to(DEVICE)
            pred = model(img)
            pred = torch.sigmoid(pred)

            ## detach and transform to numpy
            pred = pred.detach().squeeze().permute([1, 2, 0])
            pred = pred.cpu().numpy()
        
        ## convert prediction percentage confidence
        pred = pred * 100

        for i in range(num_classes):
            seg_frames[i].append(pred[:,:,i])

    return seg_frames


def load_model(yml_path, pth_path):
    with open(yml_path, "r") as f:
        yml = yaml.safe_load(f)
    
    num_classes = len(yml["DataSet"]["labels"])

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
    model = model.to(DEVICE)
    model.load_state_dict(torch.load(pth_path))
    model.eval()

    print("Seg model is loaded!")
    return model

if __name__ == "__main__":
    ROOT_BASE = "G:\\Kemal's data\\UC Berkeley studies\\High temp_OPC\\HighTemp_130404\\Kemal_Rabah"
    ROOT_SUB_DIRECTORIES = ["1st_sample", "2nd_sample", "3rd_sample", "4th_sample"]

    OUTPUT_BASE = "G:\\Kemal's data\\UC Berkeley studies\\High temp_OPC\\output"

    ## load model
    YML_PATH = "yml\\unet_smp_lovasz.yml"
    PTH_PATH = "..\\models\\unet_smp_lovasz\\unet_smp_lovasz_best.pth"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_CLASSES = 2

    model = load_model(YML_PATH, PTH_PATH)

    for subfolder in ROOT_SUB_DIRECTORIES:
        root = os.path.join(ROOT_BASE, subfolder)
        os.makedirs(os.path.join(OUTPUT_BASE, subfolder), exist_ok=True)
    
        paths = []
        paths = recursive_slect_tif(root, paths)

        paths = sorted(paths)
        for path in paths:
            if "20130128_041728_4rd_sample_T700C_10_min" in path or "20130128_044002_4rd_sample_T800C_20_min" in path:
                continue

            dirname = os.path.dirname(path)
            basename= os.path.basename(path).rstrip(".tif")
            # print(dirname)

            save_folder = dirname[len(root)+1:]
            full_save_folder = os.path.join(OUTPUT_BASE, subfolder, save_folder, basename)
            # print(full_save_folder)
            os.makedirs(full_save_folder, exist_ok=True)

            ret, frames = cv2.imreadmulti(path)
            seg_frames = process_tif_frames(frames, model, num_classes=NUM_CLASSES)
            
            for label_idx in range(NUM_CLASSES):
                os.makedirs(os.path.join(full_save_folder, f"label_{label_idx}"), exist_ok=True)
                for idx, frame in enumerate(seg_frames[label_idx]):
                    save_name = basename + "_f" + f"{idx}".zfill(5) + ".png"
                    save_frame_path = os.path.join(full_save_folder, f"label_{label_idx}", save_name)
                    cv2.imwrite(save_frame_path, frame.astype(np.uint8))
                    # save_file_path = os.path.join(save_path, f"label_{idx}", f"{name}.png")
            
            print(f"{basename} is saved in {full_save_folder}")
                
            # raise(NotImplementedError)

        print(f"Finished with subdirectory: {root}")