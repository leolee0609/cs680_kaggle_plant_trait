import torch
import numpy as np
from torch import nn
import random
import os
from datetime import datetime
import time
import math
import pandas as pd
import matplotlib.pyplot as plt
import yaml
from deepdiff import DeepDiff
from types import SimpleNamespace
from glob import glob


def sep():
    print("-"*100)

def get_timediff(time1,time2):
    minute_,second_ = divmod(time2-time1,60)
    return f"{int(minute_):02d}:{int(second_):02d}"

def current_date_time():
    # Format the current date and time as "YYYY-MM-DD HH:MM:SS"
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

def init_logger(log_file=f'train.log'):
    from logging import getLogger, INFO, FileHandler,  Formatter,  StreamHandler
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=log_file)
    handler2.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger

def write_to_summary_log(summary_log_file, message):
    with open(summary_log_file, 'a+') as file:
        file.write(f"{message}\n")

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def asMinutes(s: float):
    "Convert to minutes."
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since: float, percent: float):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (remain %s)' % (asMinutes(s), asMinutes(rs))



def load_yaml(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)
    
def simple_namespace(cfg):
    for k, v in cfg.items():
        if type(v) == dict:
            cfg[k] = SimpleNamespace(**v)
    return SimpleNamespace(**cfg)

def compare_yaml(file1, file2):
    '''
    Compare two yaml files and return the differences
    '''
    # 如果没有指定file1, 那么就和file2上一个yaml文件进行比较
    if not file1:
        all_yaml_files = sorted(glob("*.yaml"))
        if all_yaml_files.index(file2) == 0:
            print("No previous yaml file found.")
            file1 = file2
        else:
            file1 = all_yaml_files[all_yaml_files.index(file2)-1]

    yaml1 = load_yaml(file1)
    yaml2 = load_yaml(file2)
    
    def get_value_from_path(data, path):
        elements = path.strip("root").strip("[").strip("]").replace("'", "").split('][')
        for element in elements:
            try:
                if element.isdigit():
                    data = data[int(element)]
                else:
                    data = data[element]
            except (KeyError, TypeError, IndexError):
                return None
        return data

    diff = DeepDiff(yaml1, yaml2, ignore_order=True)

    # Enhance diff with actual values for added and removed items
    added_values = {}
    removed_values = {}
    for path in diff.get('dictionary_item_added', []):
        value = get_value_from_path(yaml2, path)
        added_values[path] = value

    for path in diff.get('dictionary_item_removed', []):
        value = get_value_from_path(yaml1, path)
        removed_values[path] = value

    if added_values:
        diff['dictionary_item_added_values'] = added_values
    if removed_values:
        diff['dictionary_item_removed_values'] = removed_values

    return file1, diff



def format_diffs(diffs):
    '''
    格式化diffs
    '''
    formatted_diffs = ""
    
    # 处理值变化
    for diff_type, changes in diffs.items():
        if diff_type == 'values_changed':
            for key, value in changes.items():
                path = key.split('[')[1:]
                path = [p.strip("]'") for p in path]
                path_str = " - ".join(path)
                formatted_diffs += f"{path_str}: {value['old_value']} --> {value['new_value']}\n"
        
        # 处理添加的项目
        elif diff_type == 'dictionary_item_added_values':
            for key, value in changes.items():
                path = key.split('[')[1:]
                path = [p.strip("]'") for p in path]
                path_str = " - ".join(path)
                formatted_diffs += f"[add] {path_str}: {value}\n"
        
        # 处理删除的项目
        elif diff_type == 'dictionary_item_removed_values':
            for key, value in changes.items():
                path = key.split('[')[1:]
                path = [p.strip("]'") for p in path]
                path_str = " - ".join(path)
                formatted_diffs += f"[remove] {path_str}: {value}\n"

    return formatted_diffs


def get_parameter_number(model, unit='M'):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    div = {"M": 1e6, "K": 1e3, "B": 1}[unit]
    return f'Total params: {total_num/div:.1f}{unit}; Trainable params: {trainable_num//div:.1f}{unit}'





def label_smoothing(true_distributions: torch.Tensor, smoothing=0.1):
    """
    对真实标签分布应用标签平滑。
    
    参数:
        true_distributions (torch.Tensor): 真实标签的分布张量，形状为[N, C]，其中C是类别总数。
        smoothing (float): 平滑值，默认为0.1。
        
    返回:
        torch.Tensor: 平滑后的标签分布张量，形状为[N, C]。
    """
    assert 0 <= smoothing < 1
    confidence = 1.0 - smoothing
    classes = true_distributions.size(1)
    with torch.no_grad():
        # 平滑操作
        smooth_dist = true_distributions * confidence + torch.ones_like(true_distributions) * smoothing / classes
    return smooth_dist


def compute_loss_with_label_smoothing(output, target, smoothing=0.1):
    """
    使用标签平滑计算损失。
    
    参数:
        output (torch.Tensor): 模型的预测输出，已经过log_softmax处理，形状为[N, C]。
        target (torch.Tensor): 真实标签的张量，形状为[N,]。
        classes (int): 类别总数。
        smoothing (float): 平滑值，默认为0.1。
        
    返回:
        torch.Tensor: 计算得到的损失值。
    """
    kl_loss = nn.KLDivLoss(reduction="batchmean")
    smooth_target = label_smoothing(true_distributions=target, smoothing=smoothing)
    return kl_loss(output, smooth_target)


def mixup(input, labels, clip=[0, 1]):
    """
    实现 mixup 数据增强策略。
    输入:
    - input: 输入的数据
    - labels: 真实标签
    - clip: lambda 参数的取值范围，默认为 [0, 1]

    返回:
    - input: 经过 mixup 处理后的数据
    - labels: 原始真实标签
    - shuffled_labels: 打乱顺序后的标签
    - lam: lambda 参数的值
    """
    indices = torch.randperm(input.size(0))  # 随机排列索引
    shuffled_input = input[indices]          # 根据索引打乱输入数据
    shuffled_labels = labels[indices]         # 根据索引打乱标签

    lam = np.random.uniform(clip[0], clip[1])  # 随机生成 lambda 参数
    input = input * lam + shuffled_input * (1 - lam)  # 应用 mixup 策略
    return input, labels, shuffled_labels, lam