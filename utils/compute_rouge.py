import rouge


# 计算rouge分数，   str,    str,    compute_type='f' or 'p' or 'r'
def compute_rouge(source, target, compute_type='f'):
    # 把 每 个 汉 字 用 空 格 分 开
    source, target = ' '.join(source), ' '.join(target)
    try:
        scores = rouge.Rouge().get_scores(hyps=source, refs=target)
        return {
            'rouge-1': scores[0]['rouge-1'][compute_type],
            'rouge-2': scores[0]['rouge-2'][compute_type],
            'rouge-l': scores[0]['rouge-l'][compute_type],
        }
    except ValueError:
        return {
            'rouge-1': 0.0,
            'rouge-2': 0.0,
            'rouge-l': 0.0,
        }


# 计算rouge分数，    list,    list,    compute_type='f' or 'p' or 'r'
def compute_rouges(sources, targets, compute_type='f'):
    scores = {
        'rouge-1': 0.0,
        'rouge-2': 0.0,
        'rouge-l': 0.0,
    }
    # 遍历列表计算rouge分数并累加
    for source, target in zip(sources, targets):
        score = compute_rouge(source, target, compute_type)
        for k, v in scores.items():
            scores[k] = v + score[k]
    # 输出rouge分数的平均值
    return {k: v / len(targets) for k, v in scores.items()}
