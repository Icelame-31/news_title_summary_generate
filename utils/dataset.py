import re
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torch._six import container_abcs, string_classes, int_classes
import json


# 从json文件中加载数据集，输出格式为 [('title', 'content'),('title', 'content')]
def load_data(filename):
    json_dic = json.load(open(filename, 'r', encoding="utf-8"))
    D = []
    for i, item in enumerate(json_dic):
        D.append((item.get('title').strip().replace('\n', ''), item.get('content').strip().replace('\n', '')))
    return D


# 使用分词器对数据集编码
def create_data(data, tokenizer, max_len=512, term='train'):
    datasets = []
    for title, content in data:
        text_ids = tokenizer.encode(content, max_length=max_len, truncation='only_first')
        # 如果是训练集，对标题也进行编码
        if term == 'train':
            summary_ids = tokenizer.encode(title, max_length=max_len, truncation='only_first')
            features = {'input_ids': text_ids,
                        'decoder_input_ids': summary_ids,
                        'attention_mask': [1] * len(text_ids),
                        'decoder_attention_mask': [1] * len(summary_ids)
                        }
        # 如果是测试集，只对文章内容进行编码
        elif term == 'dev':
            features = {'input_ids': text_ids,
                        'attention_mask': [1] * len(text_ids),
                        'title': title
                        }
        datasets.append(features)

    return datasets


# 使用Dataset类封装数据集
class MyDataset(Dataset):
    def __init__(self, dict_data):
        self.data = dict_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


def sequence_padding(inputs, length=None, padding=0):
    """Numpy函数，将序列padding到同一长度
    """
    if length is None:
        length = max([len(x) for x in inputs])

    pad_width = [(0, 0) for _ in np.shape(inputs[0])]
    outputs = []
    for x in inputs:
        x = x[:length]
        pad_width[0] = (0, length - len(x))
        x = np.pad(x, pad_width, 'constant', constant_values=padding)
        outputs.append(x)

    return np.array(outputs, dtype='int64')


def default_collate(batch):
    """组batch
    各个数据域分别转换为tensor，tensor第一个维度等于batch_size
    """
    np_str_obj_array_pattern = re.compile(r'[SaUO]')
    default_collate_err_msg_format = (
        "default_collate: batch must contain tensors, numpy arrays, numbers, "
        "dicts or lists; found {}")
    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
        return torch.stack(batch, 0, out=out)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        if elem_type.__name__ == 'ndarray' or elem_type.__name__ == 'memmap':
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(default_collate_err_msg_format.format(elem.dtype))
            return default_collate([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int_classes):
        return torch.tensor(batch, dtype=torch.long)
    elif isinstance(elem, string_classes):
        return batch
    elif isinstance(elem, container_abcs.Mapping):
        return {key: default_collate([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(default_collate(samples) for samples in zip(*batch)))
    elif isinstance(elem, container_abcs.Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            batch = sequence_padding(batch)
        return default_collate([default_collate(elem) for elem in batch])
    raise TypeError(default_collate_err_msg_format.format(elem_type))


# 加载数据的data_loader
def prepare_data(args, data_path, tokenizer, term='train'):
    data = load_data(data_path)  # 从json文件中加载数据集
    data = create_data(data, tokenizer, args.max_len, term)  # 使用分词器对数据集编码
    data = MyDataset(data)  # 使用Dataset类封装数据集
    data = DataLoader(data, batch_size=args.batch_size, collate_fn=default_collate)  # 对数据集分批
    return data
