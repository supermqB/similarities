from transformers import AutoModel, AutoTokenizer

# 指定模型名称
model_name = "TencentBAC/Conan-embedding-v1"

# 下载模型和分词器
model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 保存到本地
model.save_pretrained("./Conan-embedding-v1")
tokenizer.save_pretrained("./Conan-embedding-v1")