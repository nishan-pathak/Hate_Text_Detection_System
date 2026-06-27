import re
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Union
_RE_URL=re.compile(r"https?://\S+|www\.\S+")
_RE_MENTION=re.compile(r"@\w+")
_RE_HASHTAG=re.compile(r"#\w+")
_RE_NONDEV=re.compile(r"[^\u0900-\u097F\s।,?!]")
_RE_WS=re.compile(r"\s+")
_RE_DEVWORD=re.compile(r"[\u0900-\u097F]+")
def _preprocess(text):
    text=_RE_URL.sub("",text); text=_RE_MENTION.sub("",text)
    text=_RE_HASHTAG.sub("",text); text=_RE_NONDEV.sub("",text)
    return _RE_WS.sub(" ",text).strip()
def _mask(text):
    return _RE_DEVWORD.sub(lambda m:"*"*len(m.group()),text)
@dataclass
class DetectionResult:
    text:str; cleaned_text:str; label:str; is_hate:bool
    confidence:float; prob_hate:float; prob_non_hate:float; filtered_text:str
    def to_dict(self): return asdict(self)
    def __repr__(self):
        bar="█"*int(self.confidence/5)+"░"*(20-int(self.confidence/5))
        return(f"\n{'─'*45}\n  Label      : {'HATE ⚠' if self.is_hate else 'SAFE ✓'}\n"
               f"  Confidence : [{bar}] {self.confidence:.1f}%\n"
               f"  Hate prob  : {self.prob_hate:.1f}%\n  Safe prob  : {self.prob_non_hate:.1f}%\n"
               f"  Output     : {self.filtered_text}\n{'─'*45}")
class NepaliHateDetector:
    def __init__(self,model_dir,max_length=128,device=None):
        try:
            import torch
            from transformers import AutoTokenizer,AutoModelForSequenceClassification
        except ImportError as e:
            raise ImportError("Run: pip install torch transformers sentencepiece safetensors") from e
        import torch
        from transformers import AutoTokenizer,AutoModelForSequenceClassification
        self.model_dir=Path(model_dir); self.max_length=max_length
        if not self.model_dir.exists():
            raise FileNotFoundError(f"Model folder not found: {self.model_dir.resolve()}")
        self._device=torch.device(device if device else("cuda" if torch.cuda.is_available() else"cpu"))
        print(f"[NepaliHateDetector] Loading model from: {self.model_dir}")
        self._tokenizer=AutoTokenizer.from_pretrained(str(self.model_dir),use_fast=False)
        self._model=AutoModelForSequenceClassification.from_pretrained(str(self.model_dir),num_labels=2)
        self._model.to(self._device); self._model.eval()
        print(f"[NepaliHateDetector] Ready on {self._device} ✓")
    def predict(self,text):
        import torch
        if not text or not text.strip(): raise ValueError("Input text cannot be empty.")
        clean=_preprocess(text)
        if len(clean)<=1: raise ValueError("Text too short or has no Devanagari characters.")
        enc=self._tokenizer(clean,max_length=self.max_length,padding="max_length",truncation=True,return_tensors="pt")
        input_ids=enc["input_ids"].to(self._device); attention_mask=enc["attention_mask"].to(self._device)
        with torch.no_grad():
            logits=self._model(input_ids=input_ids,attention_mask=attention_mask).logits
            probs=torch.softmax(logits,dim=-1).squeeze().tolist()
        pred_id=int(torch.argmax(logits,dim=-1).item()); is_hate=pred_id==1
        confidence=round(probs[pred_id]*100,2)
        return DetectionResult(text=text,cleaned_text=clean,label="HATE" if is_hate else "NON-HATE",
            is_hate=is_hate,confidence=confidence,prob_hate=round(probs[1]*100,2),
            prob_non_hate=round(probs[0]*100,2),filtered_text=_mask(clean) if is_hate else clean)
    def predict_batch(self,texts): return [self.predict(t) for t in texts]
    def is_hate(self,text): return self.predict(text).is_hate
    def filter(self,text): return self.predict(text).filtered_text
