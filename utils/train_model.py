import os
import torch
from tqdm.auto import tqdm
from utils.compute_rouge import compute_rouges


# 训练模型
def train_model(model, adam, train_data, dev_data, tokenizer, device, args):
    # 创建模型参数保存路径文件夹
    if not os.path.exists(args.model_dir):
        os.mkdir(args.model_dir)
    # 训练模型
    best = 0  # 用于记录训练的最好的分数
    for epoch in range(args.num_epoch):

        # 设置模型为训练模式
        model.train()
        # 遍历数据集训练
        for i, item in enumerate(tqdm(train_data, desc='Epoch {}'.format(epoch))):
            item = {k: v.to(device) for k, v in item.items()}
            prob = model(**item)[0]
            mask = item['decoder_attention_mask'][:, 1:].reshape(-1).bool()
            prob = prob[:, :-1]
            prob = prob.reshape((-1, prob.size(-1)))[mask]
            labels = item['decoder_input_ids'][:, 1:].reshape(-1)[mask]
            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(prob, labels)
            if i % 100 == 0:
                print("Iter {}:  Training Loss: {}".format(i, loss.item()))
            loss.backward()
            adam.step()
            adam.zero_grad()
        # 每一轮训练结束保存模型参数
        torch.save(model, os.path.join(args.model_dir, 'model_' + str(epoch + 1)))

        # 验证模型性能，将分数最高的模型单独保存
        model.eval()
        gen_titles = []
        label_titles = []
        for item in tqdm(dev_data):
            title = item['title']
            content = {k: v.to(device) for k, v in item.items() if k != 'title'}
            # 多卡训练
            if args.data_parallel and torch.cuda.is_available():
                gen = model.module.generate(max_length=args.max_len_generate,
                                            min_length=args.min_len_generate,
                                            eos_token_id=tokenizer.sep_token_id,
                                            decoder_start_token_id=tokenizer.cls_token_id,
                                            **content)
            else:  # 单卡训练
                gen = model.generate(max_length=args.max_len_generate,
                                     min_length=args.min_len_generate,
                                     eos_token_id=tokenizer.sep_token_id,
                                     decoder_start_token_id=tokenizer.cls_token_id,
                                     **content)
            gen = tokenizer.batch_decode(gen, skip_special_tokens=True)
            gen = [item.replace(' ', '') for item in gen]
            gen_titles.extend(gen)
            label_titles.extend(title)
        # 计算生成标题的平均rouge分数
        scores = compute_rouges(gen_titles, label_titles, 'f')
        print("验证集的rouge分数为: {}".format(scores))
        # 单独保存rouge-l分数最高的模型
        rouge_l = scores['rouge-l']
        if rouge_l > best:
            best = rouge_l
            # 多卡训练参数保存
            if args.data_parallel and torch.cuda.is_available():
                torch.save(model.module, os.path.join(args.model_dir, 'summary_model'))
            else:  # 单卡训练参数保存
                torch.save(model, os.path.join(args.model_dir, 'summary_model'))
            print("epoch:{}的rouge-l分数更高，保存模型参数".format(epoch + 1))

