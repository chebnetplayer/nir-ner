import torch
import torch.nn as nn
from fastapi import FastAPI
import uvicorn
import logging
import torch.optim as optim

logging.basicConfig(level=logging.INFO)

id2label = {0: "B-TIM", 1: "I-ART", 2: "I-EVE", 3: "I-GEO", 4: "B-ART", 5: "B-EVE", 6: "O", 
            7: "I-TIM", 8: "I-PER", 9: "I-GPE", 10: "B-GEO", 11: "B-GPE", 12: "B-ORG", 
            13: "I-ORG", 14: "I-NAT", 15: "B-PER", 16: "B-NAT", 17: "UNK"}

label2id = {"B-TIM": 0, "I-ART": 1, "I-EVE": 2, "I-GEO": 3, "B-ART": 4, 
            "B-EVE": 5, "O": 6, "I-TIM": 7, "I-PER": 8, "I-GPE": 9, 
            "B-GEO": 10, "B-GPE": 11, "B-ORG": 12, "I-ORG": 13, "I-NAT": 14, 
            "B-PER": 15, "B-NAT": 16, "UNK": 17} 
vocab_len=31809
len_label2id=18
class LSTMModel(nn.Module):
    def __init__(self, input_dim = vocab_len, embedding_dim = 200, hidden_dim = 200, output_dim = len_label2id):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.lstm1 = nn.LSTM(embedding_dim, hidden_dim,  batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(hidden_dim * 2, hidden_dim, batch_first=True, bidirectional=True)
        self.lstm3 = nn.LSTM(hidden_dim * 2, hidden_dim, batch_first=True, bidirectional=True)
        self.lstm4 = nn.LSTM(hidden_dim * 2, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
    
    def forward(self, x):
        embedded = self.embedding(x)
        lstm1_out, _ = self.lstm1(embedded)
        lstm2_out, _ = self.lstm2(lstm1_out)
        lstm3_out, _ = self.lstm3(lstm2_out)
        lstm4_out, _ = self.lstm4(lstm3_out)
        tag_space = self.fc(lstm4_out)
        return tag_space

vocab = torch.load("vocab_obj.pth")

app = FastAPI()

@app.post("/predict")
def predict(text: str):
    l=[]
    model = LSTMModel()
    model.load_state_dict(torch.load("model.pth"))
    model.eval()
    model.to(torch.device("cuda:0"))
    for i in text.split(" "):
        if  vocab.__contains__(i.lower()):
            l.append(vocab[i.lower()])
        else:
            l.append(vocab["<unk>"])
    logging.info(l)
    with torch.no_grad():
        outputs = model(torch.tensor(l).to("cuda:0"))
    _, predicted = torch.max(outputs, 1)
    return [id2label[x] for x in predicted.tolist()]


@app.post("/train")
async def train_model(text: str, labels: list[str]):
    l=[]
    for label in labels:
        if label not in label2id:
            return "Incorrect labels"
            
    labels = torch.tensor([label2id[x] for x in labels]).to("cuda:0")
    model = LSTMModel()
    model.load_state_dict(torch.load("model.pth"))
    model.eval()
    model.to(torch.device("cuda:0"))
    for i in text.split(" "):
        if  vocab.__contains__(i.lower()):
            l.append(vocab[i.lower()])
        else:
            l.append(vocab["<unk>"])

    l = torch.tensor(l).to("cuda:0")

    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    optimizer.zero_grad()
    outputs = model(l.unsqueeze(0)) 
    loss = criterion(outputs.squeeze(0), labels)
    loss.backward()
    optimizer.step()

    torch.save(model.state_dict(), "model.pth")

    return "Model trained successfully"

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8888)
    
    
    