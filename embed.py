import pandas as pd
from numpy.linalg import solve, cholesky, norm
from numpy import ndarray, ones, diag, zeros, sqrt, square, delete
from typing import Tuple
from draw import vis_3d
from os.path import exists, join
from os import mkdir
from numpy import genfromtxt, savez_compressed, load
from typing import Tuple
import json

def adjust_match(path_in: str, path_out: str) -> None:
    df = pd.read_csv(path_in)
    str_category = df.columns.values[3-1]
    list_item_remove = ["Guacamole", "Nachos"]
    df = df[~df[str_category].isin(list_item_remove)]
    dict_item_replace =   {'Beans': ["Baked Beans", "Black Beans"   ], 
            'White and Brown Rice': ["White Rice",  "Brown Rice"    ]}
    list_replace = list()
    for k, v in dict_item_replace.items():
        row_replace = df[df[str_category] == k]
        df = df.drop(row_replace.index)
        row_replace = row_replace.squeeze().to_dict()
        for e in v:
            row = row_replace.copy()
            row[str_category] = e
            list_replace.append(row)
    df_replace = pd.DataFrame.from_dict(list_replace)
    df = pd.concat([df, df_replace])
    df = df.sort_values(by=str_category, key=lambda s: s.str.lower())
    df.to_csv(path_out, index = False)


def create_similar(path_data: str) -> Tuple[ndarray, list]:
    df = pd.read_csv(path_data)
    str_group, str_subgroup, str_category = df.columns.values[:3]
    df[str_category] = df[str_category].str.lower()
    df = df.sort_values(by=[str_category], ascending=True)
    list_category = [x.replace(" ", "_") for x in df[str_category].values]
    num_category = len(df)
    height_root = len([str_group, str_subgroup, str_category])
    similar = diag(ones((num_category,)))
    # pairs = [[(i, j) for j in range(i)] for i in range(1, num_category)]
    distance = lambda i, j: 0 if df[str_category][i] == df[str_category][j] \
        else 1 / height_root if df[str_subgroup][i] == df[str_subgroup][j] \
            else 2 / height_root if df[str_group][i] == df[str_group][j] \
                else 3 / height_root
    for i in range(1, num_category):
        for j in range(i):
            similar[i, j] = similar[j, i] = 1 - distance(i, j)
    return similar, list_category

def count_group(path_data: str) -> Tuple[dict, dict, dict]:
    df = pd.read_csv(path_data)
    str_group, str_subgroup, str_category = df.columns.values[:3]
    df[str_category] = df[str_category].str.lower()
    dict_count_group, dict_count_subgroup = dict(), dict()
    dict_hierarchy = dict()
    for k1, k2, v in list(zip(df[str_group].values, \
        df[str_subgroup].values, df[str_category].values)):
        if k1 not in dict_count_group.keys():
            dict_count_group[k1] = 1
        else:
            dict_count_group[k1] += 1
        if k1 not in dict_hierarchy.keys():
            dict_hierarchy[k1] = dict()
        if k2 not in dict_count_subgroup.keys():
            dict_count_subgroup[k2] = 1
        else:
            dict_count_subgroup[k2] += 1
        if k2 not in dict_hierarchy[k1].keys():
            dict_hierarchy[k1][k2] = list()
        if v not in dict_hierarchy[k1][k2]:
            dict_hierarchy[k1][k2].append(v)
    for k1, d in dict_hierarchy.items():
        for k2, v in d.items():
            d[k2] = sorted(v)
        dict_hierarchy[k1] = dict(sorted(d.items()))
    dict_hierarchy = dict(sorted(dict_hierarchy.items()))
    dict_count_group = dict(sorted(dict_count_group.items()))
    dict_count_subgroup = dict(sorted(dict_count_subgroup.items()))
    return dict_hierarchy, dict_count_group, dict_count_subgroup


def embed_sphere(similar: ndarray) -> ndarray:
    # num_category = similar.shape[0]
    # vecs_embed = zeros((num_category, num_category))
    # vecs_embed[0][0] = 1
    # for i in range(1, num_category):
    #     vecs_embed[i, :i] = solve(vecs_embed[:i, :i], similar[i, :i])
    #     vecs_embed[i, i] = sqrt(1 - sum(square(vecs_embed[i, :i])))
    # return vecs_embed
    """ similar == vecs_embed @ vecs_embed.T
    with Cholesky decomposition for the real positive-definite symmetric matrix: similar
    return matrix: vecs_embed
    note: above commented statements describe how to do the decomposition, 
    verify the results from above statements, standard Cholesky by computing Frobenius norm of difference
    >>> vecs_embed = embed_sphere(similar)
    >>> vecs_embed_cholesky = cholesky(similar)
    >>> norm(vecs_embed - vecs_embed_cholesky, ord="fro")
    6.756464008389978e-16
    """
    return cholesky(similar)

def match_category(list_category: list, list_category_hierarchy: list,
    is_VFN_1_0: bool=True) -> Tuple[list, list, dict]:
    """ return (missing categories found in `list_category`, but not in `list_category_hierarchy`,
                redundant categories found in `list_category_hierarchy`, but not in `list_category`)
    """
    # categories in `list_category`, not in `list_category_hierarchy`
    category_missing = set(list_category).difference(list_category_hierarchy)
    # categories in `list_category_hierarchy`, not in `list_category`
    category_redundant = set(list_category_hierarchy).difference(list_category)
    map_category = dict((x, x) for x in set(list_category).intersection(list_category_hierarchy))
    for v in category_missing.copy():
        _v = v.replace("_", "")
        variants = [v.rstrip("s"), v.rstrip("es"), v+"s", v+"es"]
        variants += [_v.rstrip("s"), _v.rstrip("es"), _v+"s", _v+"es"]
        for k in variants:
            if k in category_redundant:
                category_missing.remove(v)
                category_redundant.remove(k)
                map_category[k] = v
                break
    remove_replicate = set()
    for v in category_missing.copy():
        _v = v.rstrip("s")
        for k in category_redundant.copy():
            if _v in k:
                if k not in map_category.keys():
                    map_category[k] = v
                elif not isinstance(map_category[k], list):
                    map_category[k] = [map_category[k], v]
                else:
                    map_category[k].append(v)
                category_missing.remove(v)
                remove_replicate.add(k)
    for k in category_redundant.copy():
        _k = k.rstrip("s")
        remove_k = False
        for v in category_missing.copy():
            if _k in v or all(x in _k.split("_") for x in v.rstrip("s").split("_")):
                if k not in map_category.keys():
                    map_category[k] = v
                elif not isinstance(map_category[k], list):
                    map_category[k] = [map_category[k], v]
                else:
                    map_category[k].append(v)
                category_missing.remove(v)
                remove_k = True
        if remove_k:
            remove_replicate.add(k)
    category_redundant -= remove_replicate
    if is_VFN_1_0:
        map_manual =   {"whole_chicken": "turkey",
                        "quick_breads": "cornbread",
                        "frankfurter_sandwich": "hot_dog",
                        "melons": "cantaloupe",
                        "macaroni_or_noodles_with_cheese": "mac_and_cheese"}
        category_missing -= set(map_manual.values())
        category_redundant -= set(map_manual.keys())
        map_category.update(map_manual)
    return category_missing, category_redundant, dict(sorted(map_category.items()))

def embed_vec(similar: ndarray, list_category_hierarchy: list, category_redundant: set) -> dict:
    inds = [list_category_hierarchy.index(x) for x in category_redundant]
    list_t = [x for x in list_category_hierarchy if x not in category_redundant]
    _similar = delete( similar, inds, axis=0)
    _similar = delete(_similar, inds, axis=1)
    vecs_embed = embed_sphere(_similar)
    # optional: for visualization
    if not exists("embed"):
        mkdir("embed")
    vis_3d(vecs_embed, dir_embed="embed")
    return dict((c, vecs_embed[i]) for i, c in enumerate(list_t))

def match_vec(dict_vec_hierarchy: dict, map_category: dict) -> dict:
    dict_vec = dict()
    for k, v in map_category.items():
        if isinstance(v, list):
            for _v in v:
                dict_vec[_v] = dict_vec_hierarchy[k]
        else:
            dict_vec[v] = dict_vec_hierarchy[k]
    return dict(sorted(dict_vec.items()))


def save_map(fname_hierarchy: str="final_match.csv", fname_categories: str="category.txt",
            is_VFN_1_0: bool=True) -> None:
    path_hierarchy = join("hierarchy", fname_hierarchy) 
    path_category = join("label", fname_categories)
    _, list_category_hierarchy = create_similar(path_hierarchy)
    dict_category = dict(genfromtxt(path_category, delimiter=" ", dtype=None, encoding=None))
    list_category = list(dict_category.keys())
    category_missing, category_redundant, map_category = match_category(list_category, 
                                                                        list_category_hierarchy,
                                                                        is_VFN_1_0)
    if not exists("embed"):
        mkdir("embed")
    with open(join("embed", "map_category.json"), "w") as fp:
        json.dump( {"map_category": map_category, 
                    "category_missing": sorted(list(category_missing)), 
                    "category_redundant": sorted(list(category_redundant))}, fp, indent=4)


def save_embed(fname_hierarchy: str="final_match.csv", fname_map_category: str="map_category.json") -> None:
    path_hierarchy = join("hierarchy", fname_hierarchy) 
    similar, list_category_hierarchy = create_similar(path_hierarchy)
    data = json.load(open(join("embed", fname_map_category), "r"))
    map_category, category_missing, category_redundant \
        = data["map_category"], data["category_missing"], data["category_redundant"]
    dict_vec_hierarchy = embed_vec(similar, list_category_hierarchy, category_redundant)
    dict_vec = match_vec(dict_vec_hierarchy, map_category)
    savez_compressed(join("embed", "dict_vec_hierarchy.npz"), **dict_vec_hierarchy)
    savez_compressed(join("embed", "dict_vec.npz"), **dict_vec)



def load_embed():
    dict_vec_hierarchy = load(join("embed", "dict_vec_hierarchy.npz"))
    print(len(dict_vec_hierarchy.files))
    dict_vec = load(join("embed", "dict_vec.npz"))
    print(len(dict_vec.files))
    data = json.load(open(join("embed", "map_category.json"), "r"))
    map_category, category_missing, category_redundant \
        = data["map_category"], data["category_missing"], data["category_redundant"]
    print(map_category, category_missing, category_redundant)

if __name__ == "__main__":
    fname_hierarchy = "final_match_VFN_1_0.csv" # "final_match.csv"
    path_in = join("hierarchy", "final_match.csv")
    path_out = join("hierarchy", fname_hierarchy)
    adjust_match(path_in, path_out)
    save_embed(fname_hierarchy=fname_hierarchy, fname_categories="category.txt")
    path_hierarchy = path_out
    # similar, list_category_hierarchy = create_similar(path_hierarchy)
    # dict_category = dict(genfromtxt(join("label", "category.txt"), delimiter=" ", dtype=None, encoding=None))
    # list_category = list(dict_category.keys())
    # category_missing, category_redundant, map_category = match_category(list_category, list_category_hierarchy)
    # print(category_missing, str(len(category_missing)) + "/" + str(len(list_category)))
    # print(category_redundant, str(len(category_redundant)) + "/" + str(len(list_category_hierarchy)))
    # for k, v in map_category.items():
    #     print("{}: {}".format(k, v))
    # print(len(map_category))
    # dict_vec_hierarchy = embed_vec(similar, list_category_hierarchy, category_redundant)
    # dict_vec = match_vec(dict_vec_hierarchy, map_category)
    # print(dict_vec_hierarchy.keys())
    # print(len(dict_vec_hierarchy))
    # print(dict_vec.keys())
    # print(len(dict_vec))
    # print(len(dict_vec["apple"]))
    # savez_compressed(join("embed", "dict_vec_hierarchy.npz"), **dict_vec_hierarchy)
    # savez_compressed(join("embed", "dict_vec.npz"), **dict_vec)
    # with open(join("embed", "map_category.json"), "w") as fp:
    #     json.dump( {"map_category": map_category, 
    #                 "category_missing": sorted(list(category_missing)), 
    #                 "category_redundant": sorted(list(category_redundant))}, fp, indent=4)
    dict_hierarchy, dict_count_group, dict_count_subgroup = count_group(path_hierarchy)
    print(dict_hierarchy)
    print(dict_count_group)
    print(dict_count_subgroup)
    print(len(dict_count_group))
    print(len(dict_count_subgroup))
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.rcParams['mathtext.fontset'] = 'cm'
    plt.style.use('ggplot')
    xticks = [k.replace(' and', '\nand').replace(', ', ',\n') for k in dict_count_group.keys()]
    plt.bar(xticks, dict_count_group.values(), align='center', color='b')
    plt.xticks(rotation=45) # 'vertical')
    plt.grid(color='w')
    plt.title("Distribution of 74 food items in WWEIA groups")
    plt.yticks(range(0, max(dict_count_group.values())+5, 5))
    plt.tight_layout()
    plt.savefig(join("embed", "group.png"))
