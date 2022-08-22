from unicodedata import name
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'cm'
plt.style.use('ggplot')
from typing import List, Tuple
from os.path import join, exists
from os import makedirs, walk
import pandas as pd
import json

HEIGHT_ROOT = 3

def plot_metric(dir_log: str="csv_logs", name: str="model/fit") -> None:
    _, subdirs, _ = list(walk(join(dir_log, name)))[0]
    subdirs.sort()
    for subdir in subdirs:
        _plot_metric(dir_log, name, subdir)

def _plot_metric(dir_log: str="csv_logs", name: str="model/fit", subdir: str="version_0") -> None:
    path_metric = join(dir_log, name, subdir, "metrics.csv")
    dir_save = join("visualize", name, subdir)
    if not exists(dir_save):
        makedirs(dir_save)
    df = pd.read_csv(path_metric)
    str_metrics = df.columns.values
    str_metrics = [s for s in str_metrics if not (s.endswith("step") or s == "epoch")]
    dict_train, dict_val, dict_test = dict(), dict(), dict()
    for str_metric in str_metrics:
        key = str_metric.replace("_epoch", "")
        if str_metric.startswith("train"):
            key = key.replace("train_", "")
            dict_train[key] = df[str_metric].dropna().values
        elif str_metric.startswith("val"):
            key = key.replace("val_", "")
            dict_val[key] = df[str_metric].dropna().values
        elif str_metric.startswith("test"):
            key = str_metric.replace("_epoch", "")
            dict_test[key] = df[str_metric].dropna().values[0]
    if len(dict_train) != 0:
        __plot_metrics(dict_train, "train", dir_save)
    if len(dict_val) != 0:
        __plot_metrics(dict_val, "val", dir_save)
    if len(dict_test) != 0:
        if "test_num_sample" in dict_test.keys():
            num_sample = dict_test["test_num_sample" ]
            num_mistake= dict_test["test_num_mistake"]
            for k in dict_test.keys():
                if k.startswith("test_hier_dist_top"):
                    dict_test[k] *= HEIGHT_ROOT/num_sample
            dict_test["test_hier_dist_mistake"] = dict_test["test_hier_dist_top1"] * (num_sample/num_mistake)
        with open(join(dir_save, "dict_test.json"), "w") as fp:
            json.dump( dict_test, fp, indent=4)


def __plot_metrics(_dict: dict, train_val: str="train", dir_save: str="") -> None:
    dict_loss, dict_acc, dict_dist = dict(), dict(), dict()
    keys = list(_dict.keys())
    for k in keys:
        if k.startswith("loss"):
            dict_loss[k] = _dict.pop(k)
        elif k.startswith("acc"):
            dict_acc[k]  = _dict.pop(k)
        elif k.startswith("hier_dist_top"):
            dict_dist[k] = _dict.pop(k)

    fig, ax1 = plt.subplots()
    color1, color2 = 'tab:red', 'tab:blue'
    for k, v in sorted(dict_loss.items()):
        if k == "loss":
            ax1.plot(range(len(v)), v, label=r"total loss $L$", color=color1)
            ax1.set_xlim([0, len(v)-1])
        elif k == "loss_class":
            ax1.plot(range(len(v)), v, label=r"classification loss $L_{{{0}}}$".format("class"), linestyle="--", color=color1)
            ax1.legend(loc='upper left')
        elif k == "loss_cluster":
            ax1.plot(range(len(v)), v, label=r"ancillary clustering loss $L_{{{0}}}$".format("cluster"), linestyle="--", color=color2)
            ax1.legend(loc='upper left')
        elif k == "loss_corr":
            ax2 = ax1.twinx()
            ax2.plot(range(len(v)), v, label=r"ancillary correlation loss $L_{{{0}}}$".format("corr"), linestyle="--", color=color2)
            ax2.legend(loc='upper right')
            ax2.tick_params(axis='y', labelcolor=color2)
            ax2.set_ylim([0, None])
    ax1.set_xlabel("Epoch") 
    ax1.set_ylim([0, None])
    plt.grid(color='w')
    # plt.tight_layout()
    if train_val == "train":
        plt.title("loss function on training dataset")
        plt.savefig(join(dir_save, "train_loss.png"))  
    elif train_val == "val":
        plt.title("loss function on validation dataset")
        plt.savefig(join(dir_save, "val_loss.png"))  
    plt.close()

    for k, v in sorted(dict_acc.items()):
        if k == "acc":
            plt.plot(range(len(v)), v, label="accuracy at top 1")
            plt.xlim([0, len(v)-1])
        elif k.startswith("acc_top"):
            plt.plot(range(len(v)), v, label="accuracy at top {0}".format(k.replace("acc_top", "")))
    plt.xlabel("Epoch") 
    plt.ylim([0, 1])
    plt.legend()
    plt.grid(color='w')
    # plt.tight_layout()
    if train_val == "train":
        plt.title("accuracy on training dataset")
        plt.savefig(join(dir_save, "train_acc.png"))  
    elif train_val == "val":
        plt.title("accuracy on validation dataset")
        plt.savefig(join(dir_save, "val_acc.png"))  
    plt.close()

    for k, v in sorted(_dict.items()):
        plt.plot(range(len(v)), v, label=k)
    plt.xlabel("Epoch") 
    plt.xlim([0, len(list(_dict.items())[0][-1])-1])
    plt.ylim([0, 1])
    plt.legend()
    plt.grid(color='w')
    # plt.tight_layout()
    if train_val == "train":
        plt.title("metrics on training dataset")
        plt.savefig(join(dir_save, "train_metric.png"))  
    elif train_val == "val":
        plt.title("metrics on validation dataset")
        plt.savefig(join(dir_save, "val_metric.png"))  
    plt.close()

    if len(dict_dist) == 0:
        return
    num_sample = _dict["num_sample"]
    for k, v in sorted(dict_dist.items(), key=lambda t: int(t[0].replace("hier_dist_top", ""))):
        v *= HEIGHT_ROOT/num_sample
        plt.plot(range(len(v)), v, label="average hierarchical distance at top {0}".format(k.replace("hier_dist_top", "")))
    plt.xlabel("Epoch") 
    plt.xlim([0, len(list(_dict.items())[0][-1])-1])
    plt.ylim([0, None])
    plt.legend()
    plt.grid(color='w')
    # plt.tight_layout()
    if train_val == "train":
        plt.title("average hierarchical distance on training dataset")
        plt.savefig(join(dir_save, "train_dist.png"))  
    elif train_val == "val":
        plt.title("average hierarchical distance on validation dataset")
        plt.savefig(join(dir_save, "val_dist.png"))  
    plt.close()

    num_mistake = _dict["num_mistake"]
    dist_mistake = dict_dist["hier_dist_top1"]
    dist_mistake *= num_sample/num_mistake
    plt.plot(range(len(dist_mistake)), dist_mistake, label="hierarchical distance of a mistake")
    plt.xlabel("Epoch") 
    plt.xlim([0, len(list(_dict.items())[0][-1])-1])
    plt.ylim([0, None])
    plt.legend()
    plt.grid(color='w')
    if train_val == "train":
        plt.title("hierarchical distance of a mistake on training dataset")
        plt.savefig(join(dir_save, "train_dist_mistake.png"))  
    elif train_val == "val":
        plt.title("hierarchical distance of a mistake on validation dataset")
        plt.savefig(join(dir_save, "val_dist_mistake.png"))  
    plt.close()




if __name__ == "__main__":
    plot_metric(dir_log="csv_logs", name="pretrain/fit")
    plot_metric(dir_log="csv_logs", name="pretrain/test")
    plot_metric(dir_log="csv_logs", name="model/fit")
    plot_metric(dir_log="csv_logs", name="model/test")