from transformers import AutoModel, AutoTokenizer

# 指定模型名称
model_name = "shibing624/text2vec-base-chinese"

# 下载模型和分词器
model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 保存到本地
model.save_pretrained("./text2vec-base-chinese")
tokenizer.save_pretrained("./text2vec-base-chinese")