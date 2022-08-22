import torch
from torchvision import transforms
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from typing import List

class ImagePredLogger(Callback):
    def __init__(self, val_samples, num_samples=15, 
                str_categories: List[str]=['almonds', 'apple']):
        super().__init__()
        self.num_samples = num_samples
        self.val_imgs, self.val_labels = val_samples
        self.str_categories = str_categories
        self.denormalize = transforms.Compose([
            transforms.Normalize(mean=[ 0., 0., 0. ],
                                 std =[ 1/0.229, 1/0.224, 1/0.225 ]),
            transforms.Normalize(mean=[ -0.485, -0.456, -0.406 ],
                                 std =[ 1., 1., 1. ])])
        
    def on_validation_epoch_end(self, trainer, pl_module):
        # Bring the tensors to CPU
        val_imgs = self.val_imgs.to(device=pl_module.device)
        val_labels = self.val_labels.to(device=pl_module.device)
        # Get model prediction
        outputs = pl_module(val_imgs)
        logits = outputs[0] if isinstance(outputs, tuple) else outputs
        preds = torch.argmax(logits, -1)
        # Log the images as wandb Image
        index = 0
        for x, pred, y in zip(val_imgs[:self.num_samples], 
                              preds[:self.num_samples], 
                              val_labels[:self.num_samples]):
            x = self.denormalize(x)
            trainer.logger[0]\
                .experiment.add_image(
                    "Epoch:{:03d}, ".format(pl_module.current_epoch)+
                    f"Pred:{self.str_categories[pred]} {pred}, Label:{self.str_categories[y]} {y}", 
                    x, index, dataformats='CHW') # logger[0]: tb_logger
            index += self.num_samples


class ImagePredLoggerHierarchy(Callback):
    def __init__(self, val_samples, num_samples=15, 
                str_categories: List[str]=['almonds', 'apple'],
                clusters: List[int]=[1, 0]):
        super().__init__()
        self.num_samples = num_samples
        self.val_imgs, self.val_labels = val_samples
        self.str_categories = str_categories
        self.clusters = torch.as_tensor(clusters)
        self.denormalize = transforms.Compose([
            transforms.Normalize(mean=[ 0., 0., 0. ],
                                 std =[ 1/0.229, 1/0.224, 1/0.225 ]),
            transforms.Normalize(mean=[ -0.485, -0.456, -0.406 ],
                                 std =[ 1., 1., 1. ])])
        
    def on_validation_epoch_end(self, trainer, pl_module):
        # Bring the tensors to CPU
        val_imgs = self.val_imgs.to(device=pl_module.device)
        val_labels = self.val_labels.to(device=pl_module.device)
        val_labels_cluster = self.clusters[val_labels]
        # Get model prediction
        logits, logits_cluster = pl_module(val_imgs)
        preds = torch.argmax(logits, -1)
        preds_cluster = torch.argmax(logits_cluster, -1)
        # Log the images as wandb Image
        index = 0
        for x, pred, pred_cluster, y, y_cluster \
            in zip(val_imgs[:self.num_samples], 
                    preds[:self.num_samples], 
                    preds_cluster[:self.num_samples],
                    val_labels[:self.num_samples],
                    val_labels_cluster[:self.num_samples]):
            x = self.denormalize(x)
            trainer.logger[0]\
                .experiment.add_image(
                    "Epoch:{:03d}, ".format(pl_module.current_epoch)+
                    f"Pred:{self.str_categories[pred]} {pred}, Label:{self.str_categories[y]} {y}, "\
                    f"Pred_cluster:{pred_cluster}, Label_cluster:{y_cluster}", 
                    x, index, dataformats='CHW') # logger[0]: tb_logger
            index += self.num_samples


class Setting:
    def __init__(self, path_model_ckpt='models/', model_ckpt='model-{epoch:02d}-{val_loss:.2f}'):
        super().__init__()
        self.early_stop_callback = EarlyStopping(
            monitor='val_loss',
            patience=3,
            verbose=False,
            mode='min'
            )
        self.MODEL_CKPT_PATH = path_model_ckpt
        self.MODEL_CKPT = model_ckpt
        self.checkpoint_callback = ModelCheckpoint(
            monitor='val_loss',
            dirpath=self.MODEL_CKPT_PATH,
            filename=self.MODEL_CKPT,
            save_top_k=3,
            mode='min')