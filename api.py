# coding:utf-8
from flask import Flask, request
from test import generate
import torch
from utils.tokenizer import T5PegasusTokenizer
from utils.config import init_argument
import json
import time

if __name__ == '__main__':
    app = Flask(__name__)

    # 加载设置参数
    args = init_argument()
    # 设置训练设备
    device = 'cuda:' + args.device if torch.cuda.is_available() else 'cpu'
    # 加载分词器
    tokenizer = T5PegasusTokenizer.from_pretrained(args.pretrain_model)
    # 加载模型参数
    model_title = torch.load(args.model_title, map_location=device)  # 标题模型
    model_summary = torch.load(args.model_summary, map_location=device)  # 摘要模型


    # 根据id查询数据库，根据文章内容生成标题
    @app.route('/generate', methods=['GET', 'POST'])
    def get_model():
        if request.method == 'POST':
            # 获取文章内容
            content = request.form['content']
            if content == None:
                return '文章内容不可为空'
            title = generate(args, device, tokenizer, model_title, content)
            summary = generate(args, device, tokenizer, model_summary, content)
            response = json.dumps({"title": title, "summary": summary}, ensure_ascii=False)
            return response, 200, {"Content-Type": "application/json"}
        else:
            return '非法请求'


    app.run(host='127.0.0.1', debug=False, port='8999')
