import torch
from torch import nn
import torchvision.models as models
from torch.utils.data import DataLoader
# from torch.nn import DataParallel
from numpy import vstack
import gc

def get_extractor(num_classes=1000, 
            model_type="resnet",
            model_version=50, 
            pretrained=False, 
            path_pretrained=None):
    resnets = {
        18: models.resnet18, 34: models.resnet34,
        50: models.resnet50, 101: models.resnet101,
        152: models.resnet152
    }
    densenets = {
            121: models.densenet121, 161: models.densenet161,
            169: models.densenet169, 201: models.densenet201
    }
    # Using a pretrained ResNet backbone
    if (model_type is None) or (model_type == "resnet"):
        model = resnets[model_version](pretrained=pretrained)
        # Replace old FC layer with Identity so we can train our own
        # deleted: linear_size = list(self.model.children())[-1].in_features
        # added by zack
        linear_size = model.fc.in_features
        # linear_size = list(self.model.children())[-1].in_features
        # replace final layer for fine tuning
        model.fc = nn.Linear(linear_size, num_classes)
    elif model_type == "densenet":
        model = densenets[model_version](pretrained=pretrained)
        linear_size = model.classifier.in_features
        model.classifier = nn.Linear(linear_size, num_classes)
    else:
        raise Exception("model type is incorrect")
    if path_pretrained is not None:
        # added by zack, see https://pytorch.org/tutorials/recipes/recipes/saving_and_loading_a_general_checkpoint.html
        checkpoint = torch.load(path_pretrained)
        model.load_state_dict(checkpoint['state_dict'], strict=False)
    # Replace old FC layer with Identity so we can train our own
    if (model_type is None) or (model_type == "resnet"):
        model.fc = nn.Identity()
    elif model_type == "densenet":
        model.classifier = nn.Identity()
    return model


def extract_features(extractor, dataloader: DataLoader, num_classes: int=1000, gpu_single: int=3):
    device = torch.device("cuda:" + str(gpu_single))
    extractor = extractor.to(device)
    # extractor = DataParallel(extractor, device_ids=gpus) # list(range(len(gpus))))
    extractor.eval()
    dict_features = dict((str(e), []) for e in range(num_classes))
    for x, y in dataloader:
        x = x.to(device, non_blocking=True)
        y = y.cpu().numpy()
        feature = extractor(x).detach().cpu().numpy()
        for _f, _y in list(zip(feature, y)):
            dict_features[str(_y)].append(_f)
    # delete variables and free cache
    del x, y, feature, extractor
    gc.collect()
    torch.cuda.empty_cache()
    for ind in range(num_classes):
        key = str(ind)
        num = len(dict_features[key])
        dict_features[key] = vstack(dict_features[key]).reshape((num, -1))
    return dict_features


