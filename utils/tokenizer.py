import jieba
from transformers import BertTokenizer


# 从token文件中加载tokenizer编码
class T5PegasusTokenizer(BertTokenizer):

    def __init__(self, pre_tokenizer=lambda x: jieba.cut(x, HMM=False), *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pre_tokenizer = pre_tokenizer

    def _tokenize(self, text, *arg, **kwargs):
        # 结合中文特点完善的Tokenizer，基于词颗粒度的分词，如词表中未出现，再调用BERT原生Tokenizer
        split_tokens = []
        for text in self.pre_tokenizer(text):
            if text in self.vocab:
                split_tokens.append(text)
            else:
                split_tokens.extend(super()._tokenize(text))
        return split_tokens
