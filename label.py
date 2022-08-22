from os.path import dirname, join, exists
from os import makedirs, walk
from collections import defaultdict
# import sys
# sys.path.insert(0, dirname(__file__))

def generate_label(dir_image, filename_label):
    classes_images = defaultdict(list)
    _, subdirs, _ = list(walk(dir_image))[0]
    subdirs.sort()
    for category in subdirs:
        _, _, images = list(walk(join(dir_image, category)))[0]
        images.sort(key=lambda x: int(x.split('.')[0]))
        classes_images[category] = images
    with open(filename_label, 'w') as fp:
        for label, (category, images) in enumerate(classes_images.items()):
            for image in images:
                fp.write(join(category, image) + ' ' + str(label) + '\n')

def save_category(dir_image, path_category):
    _, subdirs, _ = list(walk(dir_image))[0]
    subdirs.sort()
    with open(path_category, 'w') as fp:
        for label, category in enumerate(subdirs):
            fp.write(category + ' ' + str(label) + '\n')


def get_category(subdir: str, map_category: dict):
    for k, v in map_category.items():
        if isinstance(v, list):
            if subdir in v:
                return k
        else:
            if subdir == v:
                return k

def generate_label_modified(dir_image, path_label, 
    map_category: dict, category_missing: list):
    classes_images = defaultdict(list)
    _, subdirs, _ = list(walk(dir_image))[0]
    subdirs.sort()
    categories = list(map_category.keys())
    categories.sort()
    for subdir in subdirs:
        if subdir in category_missing:
            continue
        _, _, images = list(walk(join(dir_image, subdir)))[0]
        images.sort(key=lambda x: int(x.split('.')[0]))
        classes_images[subdir] = images
    with open(path_label, 'w') as fp:
        for (subdir, images) in sorted(classes_images.items()):
            category = get_category(subdir, map_category)
            label = categories.index(category)
            for image in images:
                fp.write(join(subdir, image) + ' ' + str(label) + '\n')

def save_category_modified(map_category: dict, path_category: str):
    categories = list(map_category.keys())
    categories.sort()
    with open(path_category, 'w') as fp:
        for label, category in enumerate(categories):
            fp.write(category + ' ' + str(label) + '\n')

def generate_embed_category_label(
        dir_data: str=join("/pub2/luo333/dataset", "vfn"),
        fname_hierarchy: str="final_match.csv",
        adjust_VFN_1_0: bool=True,
        is_category_modified: bool=True, 
        use_hierarchy_WWEIA: bool=True,
        is_VFN_1_0: bool=True):
    from embed import adjust_match, save_map, save_embed
    import json
    if not exists("label"):
        makedirs("label")
    str_dataset = dir_data.split("/")[-1]
    if str_dataset in ["large_fine_food_iccv"]:
        subdir = ["Train", "Val", "Val"]
    elif str_dataset in ["CUB_200_2011", "102flowers"]:
        subdir = ["Train", "Test", "Test"]
    elif str_dataset in ["ETZH-101"]:
        subdir = ["images", "images_val", "images_test"]
    elif str_dataset in ["food-101", "stanford_cars"]:
        subdir = ["train", "test", "test"]
    elif str_dataset in ["vfn"]:
        subdir = ["vfn_train", "vfn_val", "vfn_test"]
    elif str_dataset in ["vfn_2.0"]:
        subdir = ["train", "val", "test"]
    save_category(join(dir_data, subdir[0]), join("label", "category.txt"))
    if str_dataset == "vfn" and adjust_VFN_1_0:
        fname_hierarchy_in  = fname_hierarchy
        fname_hierarchy     = "final_match_VFN_1_0.csv"
        path_in = join("hierarchy", fname_hierarchy_in)
        path_out = join("hierarchy", fname_hierarchy)
        adjust_match(path_in, path_out)
    save_map(fname_hierarchy, fname_categories="category.txt", is_VFN_1_0=is_VFN_1_0)
    if use_hierarchy_WWEIA:
        save_embed(fname_hierarchy, fname_map_category="map_category.json")
    if is_category_modified:
        path_map_category = join("embed", "map_category.json")
        map_category     = json.load(open(path_map_category, "r"))["map_category"]
        category_missing = json.load(open(path_map_category, "r"))["category_missing"]
        save_category_modified(map_category, join("label", "category.txt"))
        for dir_image, filename_label in list(zip(subdir,
                                                ["train.txt", "val.txt", "test.txt"])):
            dir_image = join(dir_data, dir_image)
            path_label = join("label", filename_label)
            generate_label_modified(dir_image, path_label, 
                                    map_category, category_missing)
    else:
        for dir_image, filename_label in list(zip(subdir,
                                            ["train.txt", "val.txt", "test.txt"])):
            dir_image = join(dir_data, dir_image)
            filename_label = join("label", filename_label)
            generate_label(dir_image, filename_label)

if __name__ == "__main__":
    dir_data        =   join("/pub2/luo333/dataset", "vfn")
    fname_hierarchy =   "final_match.csv"
    adjust_VFN_1_0       =   False
    is_category_modified =   False
    generate_embed_category_label(dir_data, fname_hierarchy, adjust_VFN_1_0, is_category_modified)