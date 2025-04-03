import numpy as np
import torch
import pickle
from .vcoco_text_label import vcoco_hoi_text_label
from .hico_text_label import hico_text_label


def generate_proj(path, dataset_type):
    pkl_path = path + '/' + 'data' + '/' + 'num_hoi_' + dataset_type + '.pkl'
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
        data = data[0]
    if dataset_type == 'vcoco':
        hoi_text_label = vcoco_hoi_text_label
        nums_obj = 81
        nums_verb = 29
    elif dataset_type == 'hico':
        hoi_text_label = hico_text_label
        nums_obj = 80
        nums_verb = 117
    else:
        assert 'dataset just for vcoco or hico'
    obj2hoi_proj = generate_o2hoi(data, hoi_text_label, nums_obj)
    verb2hoi_proj = generate_v2hoi(data, hoi_text_label, nums_verb)
    return obj2hoi_proj, verb2hoi_proj


def generate_o2hoi(data,  hoi_label, nums_obj):
    
    index_o = []
    for obj_index in range(nums_obj):
        obj_hoi = []
        for hoi_index, hoi in enumerate(hoi_label.keys()):
            if hoi[1] == obj_index:
                obj_hoi.append(hoi_index)
        index_o.append(obj_hoi)
    #print(index_o)
    o_each_num = []
    for obj_index, hoi_index in enumerate(index_o):
        num = []
        for i in hoi_index:
            j = data[i]
            num.append(j)
        o_each_num.append(num)
    o_weight = []
    for each in o_each_num:
        weight = generate_weight(each)
        o_weight.append(weight)
    # print(o_weight)
    obj2hoi_proj = torch.zeros(nums_obj, len(hoi_label))
    for index, (hoi_index, hoi_weight) in enumerate(zip(index_o, o_weight)):
        for tri_index, tri_weight in zip(hoi_index, hoi_weight):
            obj2hoi_proj[index][tri_index] = tri_weight
    # print(obj2hoi_proj)
    return obj2hoi_proj


def generate_v2hoi(data, hoi_label, nums_verb):
    index_v = []
    for verb_index in range(nums_verb):
        verb_hoi = []
        for hoi_index, hoi in enumerate(hoi_label.keys()):
            if hoi[0] == verb_index:
                verb_hoi.append(hoi_index)
        index_v.append(verb_hoi)
    # print(index_v)
    v_each_num = []
    for verb_index, hoi_index in enumerate(index_v):
        num = []
        for i in hoi_index:
            j = data[i]
            num.append(j)
        v_each_num.append(num)
    v_weight = []
    for each in v_each_num:
        weight = generate_weight(each)
        v_weight.append(weight)
    # print(v_weight)
    verb2hoi_proj = torch.zeros(nums_verb, len(hoi_label))
    for index, (hoi_index, hoi_weight) in enumerate(zip(index_v, v_weight)):
        for tri_index, tri_weight in zip(hoi_index, hoi_weight):
            verb2hoi_proj[index][tri_index] = tri_weight
    # print(verb2hoi_proj)
    return verb2hoi_proj


def list_sum(x: list):
    sum_ = 0
    for i in x:
        sum_ += i
    return sum_


def log_sigmoid(x, epsilon):
    weight = []
    for i in x:
        x_i = 1 / (1 + np.exp(-np.log(i/epsilon + 1e-4)))
        weight.append(round(x_i, 3))
    return weight


def generate_weight(x):
    if not x:
        weight = []
        return weight
    length = len(x)
    if length == 1:
        weight = [1]
    else:
        total_x = list_sum(x)
        x_max = max(x)
        x_min = min(x) + 1e-6
        div = x_max / x_min
        if div < 10:
            epsilon = total_x / length
        elif 10 <= div < 50:
            epsilon = x_max / 5
        elif div >= 50:
            epsilon = x_max / 10
        weight = log_sigmoid(x, epsilon)
    return weight


if __name__ == '__main__':
    generate_proj('data/v-coco/data/num_hoi_vcoco.pkl','vcoco')