# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: pip install gradio
"""
import gradio as gr

from similarities import BertSimilarity

sim_model = BertSimilarity()


def load_file(path):
    with open(path, 'r', encoding='utf-8') as f:
        return f.read().split('\n')


sim_model.add_corpus(load_file('data/corpus_100.txt'))


def ai_text(query):
    res = sim_model.most_similar(queries=query, topn=5)
    print(res)
    for q_id, c in enumerate(res):
        print('query:', query)
        print("search top 5:")
        print(f'\t{c}')
    res_show = '\n'.join(
        ['search top5：'] + [f'text: {k.get("corpus_doc")} score: {k.get("score"):.4f}' for k in res[0]])
    return res_show


if __name__ == '__main__':
    examples = [
        ['星巴克被嘲笑了'],
        ['西班牙失业率超过50%'],
        ['她在看书'],
        ['一个人弹琴'],
    ]
    gr.Interface(
        ai_text,
        inputs=gr.Textbox(lines=2, label="Enter Query"),
        outputs=gr.Textbox(label="Output Box"),
        title="Chinese Text Semantic Search Model",
        description="Copy or input Chinese text here. Submit and the machine will find the most similarity texts.",
        article="Link to <a href='https://github.com/shibing624/similarities'  style='color:blue;' target='_blank'>Github REPO</a>",
        examples=examples
    ).launch()
