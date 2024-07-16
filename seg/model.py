import torch
from torchvision.models.segmentation import fcn_resnet50, fcn_resnet101
from torchvision.models.segmentation import deeplabv3_resnet50, deeplabv3_mobilenet_v3_large, deeplabv3_resnet101
import segmentation_models_pytorch as smp


PREDEFINED_MODEL = {
        "fcn_resnet50":                 fcn_resnet50,
        "fcn_resnet101":                fcn_resnet101,
        "deeplabv3_resnet50":           deeplabv3_resnet50,
        "deeplabv3_mobilenet_v3_large": deeplabv3_mobilenet_v3_large,
        "deeplabv3_resnet101":          deeplabv3_resnet101
    }

class PredefinedModel(torch.nn.Module):
    """A wrapper of PyTorch predefined model.
    """
    def __init__(self, predefined_model):
        super().__init__()
        self.predefined_model = predefined_model

    def forward(self, x):
        return self.predefined_model(x)['out']

def create_predefined_model(name:str, num_classes:int)->torch.nn.Module:
    """Create a predefined segmentation model.

    Args:
        name: model name
        num_classes: num of classes in segmentation
    
    Return:
        predefined_mode: a predefined model
    """
    
    assert name in PREDEFINED_MODEL, f"Invalid model name: {name}"

    model = PREDEFINED_MODEL[name](weights=None, progress=False, aux_loss=None, num_classes=num_classes)
    return PredefinedModel(model)

def create_smp_predefined_model(model_type:str, num_classes:int, parameters:dict)->torch.nn.Module:
    """Create a predefined SMP segmentation model.

    Args:
        name: model name
        num_classes: num of classes in segmentation
        parameters: other model parameters
    
    Return:
        predefined_mode: a predefined model
    """
    
    parameters.update({"in_channels": 3})
    parameters.update({"classes": num_classes})

    if model_type == "unet":
        model = smp.Unet(**parameters)
    elif model_type == "unet++":
        model = smp.UnetPlusPlus(**parameters)
    elif model_type == "PSPNet":
        model = smp.PSPNet(**parameters)
    elif model_type == "DeepLabV3+":
        model = smp.DeepLabV3Plus(**parameters)
    else:
        raise NotImplementedError(f"Invalid SMP model type: {model_type}")

    return model