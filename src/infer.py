import sys, torch, json
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from .config import LABELS

tok = AutoTokenizer.from_pretrained("model")
model = AutoModelForSequenceClassification.from_pretrained("model").cuda().eval()

text = " ".join(sys.argv[1:]) or "type something!"
with torch.no_grad():
    scores = torch.sigmoid(model(**tok(text, return_tensors="pt").to("cuda")).logits)[0].cpu().tolist()

print(json.dumps(dict(zip(LABELS, scores)), indent=2))
