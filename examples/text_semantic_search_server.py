from fastapi import FastAPI
from pydantic import BaseModel
from similarities import BertSimilarity

app = FastAPI()

class RequestData(BaseModel):
    sentences: list
    corpus: list

@app.post("/semantic_search")
async def semantic_search(data: RequestData):
    model = BertSimilarity(model_name_or_path="./text2vec-base-chinese")
    model.add_corpus(data.corpus)
    res = model.most_similar(queries=data.sentences, topn=10)
    return {data.sentences[i]: c for i, c in enumerate(res)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=4001)
