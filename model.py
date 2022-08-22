import torch
from torch import nn
from torch.nn import functional as F
import torchvision.models as models
import pytorch_lightning as pl
# from pytorch_lightning.metrics.functional import accuracy
from torchmetrics.functional.classification.accuracy import accuracy
from torchmetrics.functional.classification import precision, recall, f1_score, specificity
# from torch.optim.lr_scheduler import CosineAnnealingLR # (optimizer, T_max, eta_min=0, last_epoch=-1, verbose=False)
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
# from kornia.augmentation import ColorJitter, RandomChannelShuffle, RandomHorizontalFlip, RandomThinPlateSpline
from numpy import genfromtxt, load, zeros # added by zack to retrieve clusters txt
import json

# class DataAugmentation(nn.Module):
#     """Module to perform data augmentation using Kornia on torch tensors."""
#     def __init__(self, apply_color_jitter: bool = False) -> None:
#         super().__init__()
#         self._apply_color_jitter = apply_color_jitter

#         self.transforms = nn.Sequential(
#             RandomHorizontalFlip(p=0.75),
#             RandomChannelShuffle(p=0.75),
#             RandomThinPlateSpline(p=0.75),
#         )
#         self.jitter = ColorJitter(0.5, 0.5, 0.5, 0.5)

#     @torch.no_grad()  # disable gradients for effiency
#     def forward(self, x: Tensor) -> Tensor:
#         x_out = self.transforms(x)  # BxCxHxW
#         if self._apply_color_jitter:
#             x_out = self.jitter(x_out)
#         return x_out



class LitModel(pl.LightningModule):
    def __init__(self, num_classes=1000,
                model_type="resnet", 
                model_version=50, 
                pretrained=True, 
                learning_rate=5e-4):
        super().__init__()
        # self.transform = DataAugmentation()  # per batch augmentation
        # log hyperparameters
        self.save_hyperparameters()
        self.__dict__.update(locals())
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
            self.model = resnets[model_version](pretrained=pretrained)
            # Replace old FC layer with Identity so we can train our own
            # deleted: linear_size = list(self.model.children())[-1].in_features
            # added by zack
            linear_size = self.model.fc.in_features
            # linear_size = list(self.model.children())[-1].in_features
            # replace final layer for fine tuning
            self.model.fc = nn.Linear(linear_size, num_classes)
        elif model_type == "densenet":
            self.model = densenets[model_version](pretrained=pretrained)
            linear_size = self.model.classifier.in_features
            self.model.classifier = nn.Linear(linear_size, num_classes)
        else:
            raise Exception("model type is incorrect")
        self.learning_rate = learning_rate
        
    
    # will be used during inference
    def forward(self, x):
        return self.model(x)

    # logic for a single training step
    def training_step(self, batch, batch_idx):
        x, y = batch
        # x_aug = self.transform(x)
        y_pred = self(x)
        loss = F.cross_entropy(y_pred, y)
        # training metrics
        preds = torch.argmax(y_pred, dim=1)
        acc = accuracy(preds, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, logger=True, sync_dist=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, logger=True, sync_dist=True)
        probs = F.softmax(y_pred, dim=1)
        self.log('train_acc_top5', accuracy(probs, y, top_k=5), on_step=True, on_epoch=True, logger=True, sync_dist=True)
        self.log('train_precision', precision(preds, y), on_step=True, on_epoch=True, logger=True, sync_dist=True)
        self.log('train_recall', recall(preds, y), on_step=True, on_epoch=True, logger=True, sync_dist=True)
        self.log('train_f1', f1_score(preds, y), on_step=True, on_epoch=True, logger=True, sync_dist=True)
        self.log('train_specificity', specificity(preds, y), on_step=True, on_epoch=True, logger=True, sync_dist=True)
        return loss

    # logic for a single validation step
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = F.cross_entropy(y_pred, y)
        # validation metrics
        preds = torch.argmax(y_pred, dim=1)
        acc = accuracy(preds, y)
        self.log('val_loss', loss, prog_bar=True, sync_dist=True)
        self.log('val_acc', acc, prog_bar=True, sync_dist=True)
        probs = F.softmax(y_pred, dim=1)
        self.log('val_acc_top5', accuracy(probs, y, top_k=5), on_step=True, on_epoch=True, logger=True, sync_dist=True)
        self.log('val_precision', precision(preds, y), on_step=True, on_epoch=True, logger=True, sync_dist=True)
        self.log('val_recall', recall(preds, y), on_step=True, on_epoch=True, logger=True, sync_dist=True)
        self.log('val_f1', f1_score(preds, y), on_step=True, on_epoch=True, logger=True, sync_dist=True)
        self.log('val_specificity', specificity(preds, y), on_step=True, on_epoch=True, logger=True, sync_dist=True)
        return loss

    # logic for a single testing step
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = F.cross_entropy(y_pred, y)
        # test metrics
        preds = torch.argmax(y_pred, dim=1)
        acc = accuracy(preds, y)
        self.log('test_loss', loss, prog_bar=True, sync_dist=True)
        self.log('test_acc', acc, prog_bar=True, sync_dist=True)
        probs = F.softmax(y_pred, dim=1)
        self.log('test_acc_top5', accuracy(probs, y, top_k=5), on_step=True, on_epoch=True, logger=True, sync_dist=True)
        self.log('test_precision', precision(preds, y), on_step=True, on_epoch=True, logger=True, sync_dist=True)
        self.log('test_recall', recall(preds, y), on_step=True, on_epoch=True, logger=True, sync_dist=True)
        self.log('test_f1', f1_score(preds, y), on_step=True, on_epoch=True, logger=True, sync_dist=True)
        self.log('test_specificity', specificity(preds, y), on_step=True, on_epoch=True, logger=True, sync_dist=True)
        return loss

    # logic for a single inference / prediction step
    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        x = batch[0] if isinstance(batch, tuple) else batch
        y_pred = self(x)
        return torch.argmax(y_pred, dim=1)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=10, max_epochs=40)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "metric_to_track",
            },
        }


class LitModelHierarchy(pl.LightningModule):
    def __init__(self, num_classes=1000, 
                num_clusters=100,
                model_type="resnet", 
                model_version=50, 
                pretrained=True, 
                path_pretrained=None, 
                path_clusters=None,
                learning_rate=5e-4):
        super().__init__()
        # save_hyperparameters(), see https://pytorch-lightning.readthedocs.io/en/latest/common/hyperparameters.html
        # self.save_hyperparameters(ignore=["path_pretrained", "path_clusters"])
        self.save_hyperparameters()
        self.__dict__.update(locals())
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
            self.model = resnets[model_version](pretrained=pretrained)
            # Replace old FC layer with Identity so we can train our own
            # deleted: linear_size = list(self.model.children())[-1].in_features
            # added by zack
            linear_size = self.model.fc.in_features
            # replace final layer for fine tuning
            self.model.fc = nn.Linear(linear_size, num_classes)
        elif model_type == "densenet":
            self.model = densenets[model_version](pretrained=pretrained)
            linear_size = self.model.classifier.in_features
            self.model.classifier = nn.Linear(linear_size, num_classes)
        else:
            raise Exception("model type is incorrect")
        
        if path_pretrained is not None:
            # added by zack, see https://pytorch.org/tutorials/recipes/recipes/saving_and_loading_a_general_checkpoint.html
            checkpoint = torch.load(path_pretrained)
            self.model.load_state_dict(checkpoint['state_dict'], strict=False)
        # Replace old FC layer with Identity so we can train our own
        # added by zack, see https://discuss.pytorch.org/t/how-to-copy-weights-from-one-model-to-another-model-instance-wise/17163
        if (model_type is None) or (model_type == "resnet"):
            self.fc1 = nn.Linear(linear_size, num_classes)
            self.fc1.weight.data.copy_(self.model.fc.weight.data)
            self.fc1.bias.data.copy_(self.model.fc.bias.data)
            self.fc2 = nn.Linear(linear_size, num_clusters)
            self.model.fc = nn.Identity()
        elif model_type == "densenet":
            self.fc1 = nn.Linear(linear_size, num_classes)
            self.fc1.weight.data.copy_(self.model.classifier.weight.data)
            self.fc1.bias.data.copy_(self.model.classifier.bias.data)
            self.fc2 = nn.Linear(linear_size, num_clusters)
            self.model.classifier = nn.Identity()

        if path_clusters == None:
            self.clusters = torch.arange(num_classes)
        else:
            clusters = genfromtxt(path_clusters, delimiter=" ", dtype=None, encoding=None)
            self.clusters = torch.as_tensor([int(row[-1]) for row in clusters])
        self.learning_rate = learning_rate
        
    
    # will be used during inference
    def forward(self, x):
        # deleted: return self.model(x)
        # added by zack
        x = self.model(x)
        return self.fc1(x), self.fc2(x)

    # logic for a single training step
    def training_step(self, batch, batch_idx):
        x, y = batch
        # x_aug = self.transform(x)
        # deleted: y_pred = self(x)
        # added by zack
        y_pred, cluster_pred = self(x)
        loss_class = F.cross_entropy(y_pred, y)
        cluster = self.clusters.to(y.device)[y]
        loss_cluster = F.cross_entropy(cluster_pred, cluster)
        loss = loss_class + loss_cluster
        # training metrics
        preds = torch.argmax(y_pred, dim=1)
        preds_cluster = torch.argmax(cluster_pred, dim=1)
        acc = accuracy(preds, y)
        acc_cluster = accuracy(preds_cluster, cluster)
        self.log('train_loss_class', loss_class, on_step=True, on_epoch=True, logger=True, sync_dist=True)
        self.log('train_loss_cluster', loss_cluster, on_step=True, on_epoch=True, logger=True, sync_dist=True)
        self.log('train_loss', loss, on_step=True, on_epoch=True, logger=True, sync_dist=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, logger=True, sync_dist=True)
        self.log('train_acc_cluster', acc_cluster, on_step=True, on_epoch=True, logger=True, sync_dist=True)
        probs = F.softmax(y_pred, dim=1)
        self.log('train_acc_top5', accuracy(probs, y, top_k=5), on_step=True, on_epoch=True, logger=True, sync_dist=True)
        self.log('train_precision', precision(preds, y), on_step=True, on_epoch=True, logger=True, sync_dist=True)
        self.log('train_recall', recall(preds, y), on_step=True, on_epoch=True, logger=True, sync_dist=True)
        self.log('train_f1', f1_score(preds, y), on_step=True, on_epoch=True, logger=True, sync_dist=True)
        self.log('train_specificity', specificity(preds, y), on_step=True, on_epoch=True, logger=True, sync_dist=True)
        return loss

    # logic for a single validation step
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred, cluster_pred = self(x)
        loss_class = F.cross_entropy(y_pred, y)
        cluster = self.clusters.to(y.device)[y]
        loss_cluster = F.cross_entropy(cluster_pred, cluster)
        loss = loss_class + loss_cluster
        # validation metrics
        preds = torch.argmax(y_pred, dim=1)
        preds_cluster = torch.argmax(cluster_pred, dim=1)
        acc = accuracy(preds, y)
        acc_cluster = accuracy(preds_cluster, cluster)
        self.log('val_loss_class', loss_class, on_step=True, on_epoch=True, logger=True, sync_dist=True)
        self.log('val_loss_cluster', loss_cluster, on_step=True, on_epoch=True, logger=True, sync_dist=True)
        self.log('val_loss', loss, on_step=True, on_epoch=True, logger=True, sync_dist=True)
        self.log('val_acc', acc, on_step=True, on_epoch=True, logger=True, sync_dist=True)
        self.log('val_acc_cluster', acc_cluster, on_step=True, on_epoch=True, logger=True, sync_dist=True)
        probs = F.softmax(y_pred, dim=1)
        self.log('val_acc_top5', accuracy(probs, y, top_k=5), on_step=True, on_epoch=True, logger=True, sync_dist=True)
        self.log('val_precision', precision(preds, y), on_step=True, on_epoch=True, logger=True, sync_dist=True)
        self.log('val_recall', recall(preds, y), on_step=True, on_epoch=True, logger=True, sync_dist=True)
        self.log('val_f1', f1_score(preds, y), on_step=True, on_epoch=True, logger=True, sync_dist=True)
        self.log('val_specificity', specificity(preds, y), on_step=True, on_epoch=True, logger=True, sync_dist=True)
        return loss

    # logic for a single testing step
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_pred, cluster_pred = self(x)
        loss_class = F.cross_entropy(y_pred, y)
        cluster = self.clusters.to(y.device)[y]
        loss_cluster = F.cross_entropy(cluster_pred, cluster)
        loss = loss_class + loss_cluster
        # test metrics
        preds = torch.argmax(y_pred, dim=1)
        preds_cluster = torch.argmax(cluster_pred, dim=1)
        acc = accuracy(preds, y)
        acc_cluster = accuracy(preds_cluster, cluster)
        self.log('test_loss_class', loss_class, on_step=True, on_epoch=True, logger=True, sync_dist=True)
        self.log('test_loss_cluster', loss_cluster, on_step=True, on_epoch=True, logger=True, sync_dist=True)
        self.log('test_loss', loss, on_step=True, on_epoch=True, logger=True, sync_dist=True)
        self.log('test_acc', acc, on_step=True, on_epoch=True, logger=True, sync_dist=True)
        self.log('test_acc_cluster', acc_cluster, on_step=True, on_epoch=True, logger=True, sync_dist=True)
        probs = F.softmax(y_pred, dim=1)
        self.log('test_acc_top5', accuracy(probs, y, top_k=5), on_step=True, on_epoch=True, logger=True, sync_dist=True)
        self.log('test_precision', precision(preds, y), on_step=True, on_epoch=True, logger=True, sync_dist=True)
        self.log('test_recall', recall(preds, y), on_step=True, on_epoch=True, logger=True, sync_dist=True)
        self.log('test_f1', f1_score(preds, y), on_step=True, on_epoch=True, logger=True, sync_dist=True)
        self.log('test_specificity', specificity(preds, y), on_step=True, on_epoch=True, logger=True, sync_dist=True)
        return loss

    # logic for a single inference / prediction step
    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        x = batch[0] if isinstance(batch, tuple) else batch
        y_pred, cluster_pred = self(x)
        return torch.argmax(y_pred, dim=1), torch.argmax(cluster_pred, dim=1)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=10, max_epochs=40)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "metric_to_track",
            },
        }




class LitModelEmbed(pl.LightningModule):
    def __init__(self, num_classes=82, 
                dimension_vec=72,
                model_type="resnet", 
                model_version=50, 
                pretrained=True, 
                path_pretrained=None, 
                path_categories=None, # "label/category.txt"
                path_map_category=None, # "embed/map_category.json"
                path_vecs=None, # "embed/dict_vec.npz",
                learning_rate=5e-4,
                lambda_corr=0.1):
        super().__init__()
        # save_hyperparameters(), see https://pytorch-lightning.readthedocs.io/en/latest/common/hyperparameters.html
        # self.save_hyperparameters(ignore=["path_pretrained", "path_clusters"])
        self.save_hyperparameters()
        self.__dict__.update(locals())
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
            self.model = resnets[model_version](pretrained=pretrained)
            # Replace old FC layer with Identity so we can train our own
            # deleted: linear_size = list(self.model.children())[-1].in_features
            # added by zack
            linear_size = self.model.fc.in_features
            # replace final layer for fine tuning
            self.model.fc = nn.Linear(linear_size, num_classes)
        elif model_type == "densenet":
            self.model = densenets[model_version](pretrained=pretrained)
            linear_size = self.model.classifier.in_features
            self.model.classifier = nn.Linear(linear_size, num_classes)
        else:
            raise Exception("model type is incorrect")
        if path_pretrained is not None:
            # added by zack, see https://pytorch.org/tutorials/recipes/recipes/saving_and_loading_a_general_checkpoint.html
            checkpoint = torch.load(path_pretrained)
            self.model.load_state_dict(checkpoint['state_dict'], strict=False)
        # Replace old FC layer with Identity so we can train our own
        # added by zack, see https://discuss.pytorch.org/t/how-to-copy-weights-from-one-model-to-another-model-instance-wise/17163
        if (model_type is None) or (model_type == "resnet"):
            self.fc1 = nn.Linear(linear_size, num_classes)
            self.fc1.weight.data.copy_(self.model.fc.weight.data)
            self.fc1.bias.data.copy_(self.model.fc.bias.data)
            self.fc2 = nn.Linear(linear_size, dimension_vec)
            self.model.fc = nn.Identity()
        elif model_type == "densenet":
            self.fc1 = nn.Linear(linear_size, num_classes)
            self.fc1.weight.data.copy_(self.model.classifier.weight.data)
            self.fc1.bias.data.copy_(self.model.classifier.bias.data)
            self.fc2 = nn.Linear(linear_size, dimension_vec)
            self.model.classifier = nn.Identity()
        if path_categories == None or path_map_category == None:
            raise ValueError("`path_map_category` or `path_category` should have valid pathes")
        self.categories = genfromtxt(path_categories, delimiter=" ", dtype=None, encoding=None)
        self.category_missing = json.load(open(path_map_category, "r"))["category_missing"]
        if path_vecs == None:
            raise ValueError("`path_vecs` should be a valid path")
        dict_vec = load(path_vecs)
        vecs_embed = zeros((num_classes, dimension_vec))
        weight_category = zeros((num_classes,))
        for k, i in [(t[0], t[1]) for t in self.categories]:
            if k not in self.category_missing:
                vecs_embed[i] = dict_vec[k]
                weight_category[i] = 1.
        self.vecs_embed = torch.as_tensor(vecs_embed)
        self.weight_category = torch.as_tensor(weight_category)
        self.learning_rate = learning_rate
        self.lambda_corr   = lambda_corr
        
    
    # will be used during inference
    def forward(self, x):
        # deleted: return self.model(x)
        # added by zack
        x = self.model(x)
        return self.fc1(x), self.fc2(x)

    # logic for a single training step
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred, vec_pred = self(x)
        loss_class = F.cross_entropy(y_pred, y)
        vec = self.vecs_embed.to(y.device)[y]
        weight = self.weight_category.to(y.device)[y]
        loss_corr = (weight * (1-(F.normalize(vec_pred, dim=1) * vec).sum(1))).sum(0)
        loss = loss_class + self.lambda_corr * loss_corr
        # training metrics
        preds = torch.argmax(y_pred, dim=1)
        acc = accuracy(preds, y)
        self.log('train_loss_class', loss_class, on_step=True, on_epoch=True, logger=True, sync_dist=True)
        self.log('train_loss_corr', loss_corr, on_step=True, on_epoch=True, logger=True, sync_dist=True)
        self.log('train_loss', loss, on_step=True, on_epoch=True, logger=True, sync_dist=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, logger=True, sync_dist=True)
        probs = F.softmax(y_pred, dim=1)
        self.log('train_acc_top5', accuracy(probs, y, top_k=5), on_step=True, on_epoch=True, logger=True, sync_dist=True)
        self.log('train_precision', precision(preds, y), on_step=True, on_epoch=True, logger=True, sync_dist=True)
        self.log('train_recall', recall(preds, y), on_step=True, on_epoch=True, logger=True, sync_dist=True)
        self.log('train_f1', f1_score(preds, y), on_step=True, on_epoch=True, logger=True, sync_dist=True)
        self.log('train_specificity', specificity(preds, y), on_step=True, on_epoch=True, logger=True, sync_dist=True)
        # average hierarchical distance @ k=1, 5, 10, 20
        preds_top5  = torch.topk(y_pred, 5 )[-1]
        preds_top10 = torch.topk(y_pred, 10)[-1]
        preds_top20 = torch.topk(y_pred, 20)[-1]
        vec_top1  = self.vecs_embed.to(y.device)[preds]
        vec_top5  = self.vecs_embed.to(y.device)[preds_top5 ]
        vec_top10 = self.vecs_embed.to(y.device)[preds_top10]
        vec_top20 = self.vecs_embed.to(y.device)[preds_top20]
        id_mistake = torch.ne(preds, y) # not equal
        hier_dist_top1      = (weight * (1-(vec_top1 * vec).sum(-1))).sum()
        hier_dist_top5      = (weight.unsqueeze(dim=1) * (1-(vec_top5  * vec.unsqueeze(dim=1)).sum(-1))).sum() / 5
        hier_dist_top10     = (weight.unsqueeze(dim=1) * (1-(vec_top10 * vec.unsqueeze(dim=1)).sum(-1))).sum() / 10
        hier_dist_top20     = (weight.unsqueeze(dim=1) * (1-(vec_top20 * vec.unsqueeze(dim=1)).sum(-1))).sum() / 20
        self.log('train_hier_dist_top1'     , hier_dist_top1    , on_step=True, on_epoch=True, logger=True, sync_dist=True)
        self.log('train_hier_dist_top5'     , hier_dist_top5    , on_step=True, on_epoch=True, logger=True, sync_dist=True)
        self.log('train_hier_dist_top10'    , hier_dist_top10   , on_step=True, on_epoch=True, logger=True, sync_dist=True)
        self.log('train_hier_dist_top20'    , hier_dist_top20   , on_step=True, on_epoch=True, logger=True, sync_dist=True)
        self.log('train_num_mistake', (id_mistake * weight).sum(), on_step=True, on_epoch=True, logger=True, sync_dist=True)
        self.log('train_num_sample',   weight.sum(), on_step=True, on_epoch=True, logger=True, sync_dist=True)
        return loss

    # logic for a single validation step
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred, vec_pred = self(x)
        loss_class = F.cross_entropy(y_pred, y)
        vec = self.vecs_embed.to(y.device)[y]
        weight = self.weight_category.to(y.device)[y]
        loss_corr = (weight * (1-(F.normalize(vec_pred, dim=1) * vec).sum(1))).sum(0)
        loss = loss_class + self.lambda_corr * loss_corr
        # validation metrics
        preds = torch.argmax(y_pred, dim=1)
        acc = accuracy(preds, y)
        self.log('val_loss_class', loss_class, on_step=True, on_epoch=True, logger=True, sync_dist=True)
        self.log('val_loss_corr', loss_corr, on_step=True, on_epoch=True, logger=True, sync_dist=True)
        self.log('val_loss', loss, on_step=True, on_epoch=True, logger=True, sync_dist=True)
        self.log('val_acc', acc, on_step=True, on_epoch=True, logger=True, sync_dist=True)
        probs = F.softmax(y_pred, dim=1)
        self.log('val_acc_top5', accuracy(probs, y, top_k=5), on_step=True, on_epoch=True, logger=True, sync_dist=True)
        self.log('val_precision', precision(preds, y), on_step=True, on_epoch=True, logger=True, sync_dist=True)
        self.log('val_recall', recall(preds, y), on_step=True, on_epoch=True, logger=True, sync_dist=True)
        self.log('val_f1', f1_score(preds, y), on_step=True, on_epoch=True, logger=True, sync_dist=True)
        self.log('val_specificity', specificity(preds, y), on_step=True, on_epoch=True, logger=True, sync_dist=True)
        # average hierarchical distance @ k=1, 5, 10, 20
        preds_top5  = torch.topk(y_pred, 5 )[-1]
        preds_top10 = torch.topk(y_pred, 10)[-1]
        preds_top20 = torch.topk(y_pred, 20)[-1]
        vec_top1  = self.vecs_embed.to(y.device)[preds]
        vec_top5  = self.vecs_embed.to(y.device)[preds_top5 ]
        vec_top10 = self.vecs_embed.to(y.device)[preds_top10]
        vec_top20 = self.vecs_embed.to(y.device)[preds_top20]
        id_mistake = torch.ne(preds, y) # not equal
        hier_dist_top1      = (weight * (1-(vec_top1 * vec).sum(-1))).sum()
        hier_dist_top5      = (weight.unsqueeze(dim=1) * (1-(vec_top5  * vec.unsqueeze(dim=1)).sum(-1))).sum() / 5
        hier_dist_top10     = (weight.unsqueeze(dim=1) * (1-(vec_top10 * vec.unsqueeze(dim=1)).sum(-1))).sum() / 10
        hier_dist_top20     = (weight.unsqueeze(dim=1) * (1-(vec_top20 * vec.unsqueeze(dim=1)).sum(-1))).sum() / 20
        self.log('val_hier_dist_top1'       , hier_dist_top1    , on_step=True, on_epoch=True, logger=True, sync_dist=True)
        self.log('val_hier_dist_top5'       , hier_dist_top5    , on_step=True, on_epoch=True, logger=True, sync_dist=True)
        self.log('val_hier_dist_top10'      , hier_dist_top10   , on_step=True, on_epoch=True, logger=True, sync_dist=True)
        self.log('val_hier_dist_top20'      , hier_dist_top20   , on_step=True, on_epoch=True, logger=True, sync_dist=True)
        self.log('val_num_mistake', (id_mistake * weight).sum(), on_step=True, on_epoch=True, logger=True, sync_dist=True)
        self.log('val_num_sample' ,  weight.sum(), on_step=True, on_epoch=True, logger=True, sync_dist=True)
        return loss

    # logic for a single testing step
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_pred, vec_pred = self(x)
        loss_class = F.cross_entropy(y_pred, y)
        vec = self.vecs_embed.to(y.device)[y]
        weight = self.weight_category.to(y.device)[y]
        loss_corr = (weight * (1-(F.normalize(vec_pred, dim=1) * vec).sum(1))).sum(0)
        loss = loss_class + self.lambda_corr * loss_corr
        # test metrics
        preds = torch.argmax(y_pred, dim=1)
        acc = accuracy(preds, y)
        self.log('test_loss_class', loss_class, on_step=True, on_epoch=True, logger=True, sync_dist=True)
        self.log('test_loss_corr', loss_corr, on_step=True, on_epoch=True, logger=True, sync_dist=True)
        self.log('test_loss', loss, on_step=True, on_epoch=True, logger=True, sync_dist=True)
        self.log('test_acc', acc, on_step=True, on_epoch=True, logger=True, sync_dist=True)
        probs = F.softmax(y_pred, dim=1)
        self.log('test_acc_top5', accuracy(probs, y, top_k=5), on_step=True, on_epoch=True, logger=True, sync_dist=True)
        self.log('test_precision', precision(preds, y), on_step=True, on_epoch=True, logger=True, sync_dist=True)
        self.log('test_recall', recall(preds, y), on_step=True, on_epoch=True, logger=True, sync_dist=True)
        self.log('test_f1', f1_score(preds, y), on_step=True, on_epoch=True, logger=True, sync_dist=True)
        self.log('test_specificity', specificity(preds, y), on_step=True, on_epoch=True, logger=True, sync_dist=True)
        # average hierarchical distance @ k=1, 5, 10, 20
        preds_top5  = torch.topk(y_pred, 5 )[-1]
        preds_top10 = torch.topk(y_pred, 10)[-1]
        preds_top20 = torch.topk(y_pred, 20)[-1]
        vec_top1  = self.vecs_embed.to(y.device)[preds]
        vec_top5  = self.vecs_embed.to(y.device)[preds_top5 ]
        vec_top10 = self.vecs_embed.to(y.device)[preds_top10]
        vec_top20 = self.vecs_embed.to(y.device)[preds_top20]
        id_mistake = torch.ne(preds, y) # not equal
        hier_dist_top1   = (weight * (1-(vec_top1 * vec).sum(-1))).sum()
        hier_dist_top5   = (weight.unsqueeze(dim=1) * (1-(vec_top5  * vec.unsqueeze(dim=1)).sum(-1))).sum() / 5
        hier_dist_top10  = (weight.unsqueeze(dim=1) * (1-(vec_top10 * vec.unsqueeze(dim=1)).sum(-1))).sum() / 10
        hier_dist_top20  = (weight.unsqueeze(dim=1) * (1-(vec_top20 * vec.unsqueeze(dim=1)).sum(-1))).sum() / 20
        self.log('test_hier_dist_top1'      , hier_dist_top1    , on_step=True, on_epoch=True, logger=True, sync_dist=True)
        self.log('test_hier_dist_top5'      , hier_dist_top5    , on_step=True, on_epoch=True, logger=True, sync_dist=True)
        self.log('test_hier_dist_top10'     , hier_dist_top10   , on_step=True, on_epoch=True, logger=True, sync_dist=True)
        self.log('test_hier_dist_top20'     , hier_dist_top20   , on_step=True, on_epoch=True, logger=True, sync_dist=True)
        self.log('test_num_mistake', (id_mistake * weight).sum(), on_step=True, on_epoch=True, logger=True, sync_dist=True)
        self.log('test_num_sample' , weight.sum(), on_step=True , on_epoch=True, logger=True, sync_dist=True)
        return loss


    # logic for a single inference / prediction step
    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        x = batch[0] if isinstance(batch, tuple) else batch
        y_pred, vec_pred = self(x)
        return torch.argmax(y_pred, dim=1), F.normalize(vec_pred, dim=1)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=10, max_epochs=40)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "metric_to_track",
            },
        }