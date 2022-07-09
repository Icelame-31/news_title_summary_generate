import torch
from transformers import MT5ForConditionalGeneration
from utils.config import init_argument
from utils.tokenizer import T5PegasusTokenizer
from utils.dataset import prepare_data
from utils.train_model import train_model

if __name__ == '__main__':
    # 加载设置参数
    args = init_argument()
    # 设置训练设备
    device = 'cuda:' + args.device if torch.cuda.is_available() else 'cpu'
    # 加载分词器
    tokenizer = T5PegasusTokenizer.from_pretrained(args.pretrain_model)
    # 加载训练数据集和验证数据集
    train_data = prepare_data(args, args.train_data, tokenizer, term='train')
    dev_data = prepare_data(args, args.dev_data, tokenizer, term='dev')
    # 加载预训练模型
    model = MT5ForConditionalGeneration.from_pretrained(args.pretrain_model).to(device)
    # 多卡训练
    if args.data_parallel and torch.cuda.is_available():
        device_ids = range(torch.cuda.device_count())
        model = torch.nn.DataParallel(model, device_ids=device_ids)
    # 设置优化器
    adam = torch.optim.Adam(model.parameters(), lr=args.lr)
    # 训练模型
    train_model(model, adam, train_data, dev_data, tokenizer, device, args)
    print("训练结束！")