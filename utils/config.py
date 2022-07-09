import argparse


# 初始化参数
def init_argument():
    parser = argparse.ArgumentParser(description='智能创作平台标题摘要生成模型')

    parser.add_argument('--train_data', default='./data/train.json', help='训练数据集路径')
    parser.add_argument('--dev_data', default='./data/dev.json', help='验证数据集路径')
    parser.add_argument('--model_title', default='./saved_model/model_title', help='标题生成的模型参数')
    parser.add_argument('--model_summary', default='./saved_model/model_summary', help='摘要生成的模型参数')
    parser.add_argument('--pretrain_model', default='./t5_pegasus_pretrain', help='模型的预训练参数路径')
    parser.add_argument('--model_dir', default='./saved_model', help='模型参数的保存路径')
    parser.add_argument('--device', default='0', help='默认使用的gpu编号')

    parser.add_argument('--num_epoch', default=50, help='训练的epoch轮数')
    parser.add_argument('--batch_size', default=8, help='batch size')
    parser.add_argument('--lr', default=2e-4, help='学习率')
    parser.add_argument('--data_parallel', default=False, help='是否使用多卡训练')
    parser.add_argument('--max_len', default=1024, help='输入的最大长度')
    parser.add_argument('--max_len_generate', default=80, help='输出的最大长度')

    args = parser.parse_args()
    return args
