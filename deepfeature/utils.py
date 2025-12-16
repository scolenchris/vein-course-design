import sys
import torch
import numpy as np


def get_device():
    """获取计算设备"""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return device


class Logger(object):
    """日志记录器，同时输出到控制台和文件"""

    def __init__(self, filename="default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.log.flush()


def data_read(dir_path):
    """读取特定格式的数据"""
    with open(dir_path, "r") as f:
        raw_data = f.read()
        data = raw_data[1:-1].split(", ")
    return np.asarray(data, float)
