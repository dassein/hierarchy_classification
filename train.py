from data import Image_DataModule
from model import LitModel, LitModelHierarchy
from log import ImagePredLogger, ImagePredLoggerHierarchy, Setting
from feature import get_extractor, extract_features
from utils import get_dict_feature, estimate_distrib, compute_similar, cluster_category, \
    save_cluster, get_num_cluster, get_clusters, get_str_categories
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from os import walk, makedirs
from os.path import join, exists
from numpy import savez_compressed
import warnings
warnings.filterwarnings("ignore", category=UserWarning) # suppress pytorch UserWarning
import argparse

'''stage 1: pretrain on resnet'''
def stage1(dm: Image_DataModule, 
            load_checkpoint=True,
            fname_checkpoint=None,
            fname_categories="category.txt",
            num_classes=1000,
            model_type="resnet",
            model_version=50,
            learning_rate=2e-4,
            max_epochs=10, 
            gpus=[0, 1, 2, 3]):
    # Samples required by the custom ImagePredLogger callback to log image predictions.
    val_samples = next(iter(dm.val_dataloader()))
    # Init our model
    setting = Setting(path_model_ckpt="pretrains/", model_ckpt='pretrain-{epoch:02d}-{val_loss:.2f}')
    if load_checkpoint:
        path_checkpoint = setting.MODEL_CKPT_PATH + fname_checkpoint
        model = LitModel.load_from_checkpoint(path_checkpoint)
    else:
        model = LitModel(num_classes=num_classes, 
                        model_type=model_type,
                        model_version=model_version, 
                        pretrained=True, 
                        learning_rate=learning_rate)
    # Initialize tensorboard logger
    tb_logger = TensorBoardLogger('tb_logs', name=join("pretrain", dm.stage))
    csv_logger = CSVLogger("csv_logs", name=join("pretrain", dm.stage))
    # Initialize a trainer
    trainer = pl.Trainer(max_epochs=max_epochs,
                        precision=16,
                        gpus=gpus, 
                        # strategy="ddp_spawn",
                        strategy="ddp_find_unused_parameters_false",
                        logger=[tb_logger, csv_logger],
                        callbacks=[TQDMProgressBar(refresh_rate=20),
                                   setting.early_stop_callback,
                                   setting.checkpoint_callback,
                                   ImagePredLogger(
                                        val_samples, 
                                        num_samples=5,
                                        str_categories=get_str_categories(join("label", fname_categories)))],
                        deterministic=True) # ensure reproducibility for each run, see https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html
    # Train the model
    trainer.fit(model, dm)
    # Evaluate the model on the held out test set ⚡⚡
    # trainer.test()


def stage2(dm: Image_DataModule,
            fname_checkpoint="pretrain-epoch=05-val_loss=0.83.ckpt",
            num_classes=82,
            model_type="resnet",
            model_version=50,
            fname_features="features.npz", 
            gpu_single=3):
    if not exists("cluster"):
        makedirs("cluster")
    path_pretrained = join("pretrains", fname_checkpoint) if fname_checkpoint is not None else None
    extractor = get_extractor(num_classes=num_classes,
                                model_type=model_type,
                                model_version=model_version,
                                pretrained=True,
                                path_pretrained=path_pretrained)
    dict_features = extract_features(extractor=extractor, 
                                    dataloader=dm.train_dataloader(), 
                                    num_classes=num_classes,
                                    gpu_single=gpu_single)
    savez_compressed(join("cluster", fname_features), **dict_features)


def stage3(fname_features="features.npz",
            fname_categories="category.txt",
            fname_clusters="cluster.txt"):
    dict_features = get_dict_feature(join("cluster", fname_features))
    distrib = estimate_distrib(dict_features)
    similar = compute_similar(distrib)
    clusters = cluster_category(similar)
    path_categories = join("label", fname_categories)
    path_clusters = join("cluster", fname_clusters)
    save_cluster(clusters, path_categories, path_clusters)
    

def stage4(dm: Image_DataModule,
            pretrained=True, 
            fname_pretrain="pretrain-epoch=05-val_loss=0.83.ckpt",
            load_checkpoint=True,
            fname_checkpoint="model-epoch=05-val_loss=0.83.ckpt",
            fname_categories="category.txt",
            fname_clusters="cluster.txt",
            num_classes=82,
            num_clusters=11,
            model_type="resnet",
            model_version=50,
            learning_rate=1e-4,
            max_epochs=50,
            gpus=[0, 1, 2, 3]):
    # Samples required by the custom ImagePredictionLogger callback to log image predictions.
    val_samples = next(iter(dm.val_dataloader()))
    # Init our model
    setting = Setting(path_model_ckpt='models/', model_ckpt='model-{epoch:02d}-{val_loss:.2f}')
    if load_checkpoint and (fname_checkpoint is not None):
        path_checkpoint = setting.MODEL_CKPT_PATH + fname_checkpoint
        model = LitModelHierarchy.load_from_checkpoint(path_checkpoint)
    else:
        path_pretrained = None if fname_pretrain is None \
            else join("pretrains", fname_pretrain)
        path_clusters = None if fname_clusters is None \
            else join("cluster", fname_clusters)
        model = LitModelHierarchy(num_classes=num_classes,
                                num_clusters=num_clusters,
                                model_type=model_type, 
                                model_version=model_version, 
                                pretrained=pretrained, 
                                path_pretrained=path_pretrained,
                                path_clusters=path_clusters,
                                learning_rate=learning_rate)
    # Initialize tensorboard logger
    tb_logger = TensorBoardLogger('tb_logs', name=join("model", dm.stage))
    csv_logger = CSVLogger("csv_logs", name=join("model", dm.stage))
    # Initialize a trainer
    trainer = pl.Trainer(max_epochs=max_epochs,
                        precision=16,
                        gpus=gpus, 
                        # strategy="ddp_spawn",
                        strategy="ddp_find_unused_parameters_false",
                        logger=[tb_logger, csv_logger],
                        callbacks=[TQDMProgressBar(refresh_rate=20),
                                   setting.early_stop_callback,
                                   setting.checkpoint_callback,
                                   ImagePredLoggerHierarchy(
                                        val_samples, 
                                        num_samples=5,
                                        str_categories=get_str_categories(join("label", fname_categories)),
                                        clusters=get_clusters(join("cluster", fname_clusters)))],
                        deterministic=True) # ensure reproducibility for each run, see https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html
    # Train the model
    trainer.fit(model, dm)


def stage5(dm: Image_DataModule,
            fname_pretrain="pretrain-epoch=05-val_loss=0.83.ckpt",
            fname_model="model-epoch=05-val_loss=0.83.ckpt",
            gpus=[0, 1, 2, 3]):
    dm.setup(stage="test")
    path_pretrained = join("pretrains", fname_pretrain)
    path_model = join("models", fname_model)
    pretrained = LitModel.load_from_checkpoint(path_pretrained)
    model = LitModelHierarchy.load_from_checkpoint(path_model)
    # Initialize tensorboard logger
    tb_loggers = [TensorBoardLogger('tb_logs', name=join(x, dm.stage)) for x in ["pretrain", "model"]]
    csv_loggers = [CSVLogger("csv_logs", name=join(x, dm.stage)) for x in ["pretrain", "model"]]
    # Initialize a trainer
    trainers = [pl.Trainer(precision=16,
                        gpus=gpus, 
                        # strategy="ddp_spawn",
                        strategy="ddp_find_unused_parameters_false",
                        logger=[tb_logger, csv_logger],
                        callbacks=[TQDMProgressBar(refresh_rate=20)],
                        deterministic=True)
                for tb_logger, csv_logger in list(zip(tb_loggers, csv_loggers))] 
                # ensure reproducibility for each run, see https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html
    dict_result_pretrain = trainers[0].test(pretrained, dm)[0]
    dict_result_model    = trainers[1].test(model, dm)[0]
    # print(dict_result_pretrain)
    # print(dict_result_model)

if __name__ == "__main__":
    # hyperparams
    num_classes     = 82
    model_type      = "densenet" # "resnet"
    model_version   = 121 # 50
    Height, Width   = 224, 224
    batch_size      = 32 # 64
    learning_rate_pre   = 2e-4
    max_epochs_pre      = 50
    learning_rate   = 2e-4
    max_epochs      = 200
    gpus            = [1, 2, 3]
    gpu_single      = gpus[-1]
    fname_categories    = "category.txt"
    fname_clusters      = "cluster.txt"
    fname_features      = "features.npz"
    seed_run        = 42
    # ensure reproducibility for each run, set seeds for numpy, torch, python.random and PYTHONHASHSEED
    # see https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html
    pl.seed_everything(seed_run, workers=True) 
    # prepare data pipeline
    # dir_data = "/pub2/luo333/dataset/large_fine_food_iccv"
    dir_data = "/pub2/luo333/dataset/vfn"
    dm = Image_DataModule(batch_size=batch_size, stage="fit", 
                          dir_data=dir_data, 
                          H=Height, W=Width)
    dm.prepare_data()
    dm.setup(stage="fit")
    # parse the stage id for training process
    parser = argparse.ArgumentParser(description='PyTorch Hierarchical Based Classification')
    parser.add_argument('--stage', type=int, default=4,
                        help='stage id for training (choose from [1, 2, 3, 4])')
    args = parser.parse_args()
    if args.stage not in [1, 2, 3, 4, 5]:
        raise ValueError("stage must be one of [1, 2, 3, 4]")
    elif args.stage == 1:
        stage1(dm, 
            load_checkpoint=False,
            fname_checkpoint=None,
            fname_categories=fname_categories,
            num_classes=num_classes,
            model_type=model_type,
            model_version=model_version,
            learning_rate=learning_rate_pre,
            max_epochs=max_epochs_pre,
            gpus=gpus)
    elif args.stage == 2:
        _, _, pretrains = list(walk("pretrains"))[0]
        pretrains.sort(key=lambda x: float(x.split('=')[-1][:-5].split('-')[0])) # sort with val_loss
        # print(pretrains[0]) # example: "pretrain-epoch=05-val_loss=0.83.ckpt"
        stage2(dm,
                fname_checkpoint=pretrains[0],
                num_classes=num_classes,
                model_type=model_type,
                model_version=model_version,
                fname_features=fname_features,
                gpu_single=gpu_single)
    elif args.stage == 3:
        print("stage 3: organize categories into clusters\n")
        stage3(fname_features=fname_features,
                fname_categories=fname_categories,
                fname_clusters=fname_clusters)
    elif args.stage == 4:
        _, _, pretrains = list(walk("pretrains"))[0]
        pretrains.sort(key=lambda x: float(x.split('=')[-1][:-5].split('-')[0])) # sort with val_loss
        # print(pretrains[0]) # example: "pretrain-epoch=05-val_loss=0.83.ckpt"     
        num_clusters = get_num_cluster(join("cluster", fname_clusters))
        # print(num_clusters)
        stage4(dm,
                pretrained=True, 
                fname_pretrain=pretrains[0],
                load_checkpoint=False,
                fname_checkpoint=None,
                fname_categories=fname_categories,
                fname_clusters=fname_clusters,
                num_classes=num_classes,
                num_clusters=num_clusters,
                model_type=model_type,
                model_version=model_version,
                learning_rate=learning_rate,
                max_epochs=max_epochs,
                gpus=gpus)
    else: # args.stage == 5
        _, _, pretrains = list(walk("pretrains"))[0]
        pretrains.sort(key=lambda x: float(x.split('=')[-1][:-5].split('-')[0])) # sort with val_loss
        _, _, models = list(walk("models"))[0]
        models.sort(key=lambda x: float(x.split('=')[-1][:-5].split('-')[0])) # sort with val_loss
        stage5(dm, fname_pretrain=pretrains[0], fname_model=models[0], gpus=gpus)
    # run it with: python -W ignore train.py --stage 1 (or 2, 3, 4), to ignore ALL the warnings
    # run some stages in the same time with: source train.sh
    # check logger with: tensorboard --logdir ./tb_logs/train_model --port 6006
    # ps -ef | grep tensorboard, pid is 1st number of result: luo333   2402878 2355289
    # kill -9 2402878