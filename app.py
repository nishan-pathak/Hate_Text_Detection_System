import re,os,sys,time,threading,webbrowser,argparse
from pathlib import Path
_RE_URL=re.compile(r"https?://\S+|www\.\S+")
_RE_MENTION=re.compile(r"@\w+")
_RE_HASHTAG=re.compile(r"#\w+")
_RE_NONDEV=re.compile(r"[^\u0900-\u097F\s।,?!]")
_RE_WS=re.compile(r"\s+")
_RE_DEVWORD=re.compile(r"[\u0900-\u097F]+")
def _preprocess(text):
    text=_RE_URL.sub("",text);text=_RE_MENTION.sub("",text)
    text=_RE_HASHTAG.sub("",text);text=_RE_NONDEV.sub("",text)
    return _RE_WS.sub(" ",text).strip()
def _mask(text):
    return _RE_DEVWORD.sub(lambda m:"*"*len(m.group()),text)
def create_app(model_dir,port=5001):
    from flask import Flask,request,jsonify,render_template
    from flask_cors import CORS
    import torch
    from transformers import AutoTokenizer,AutoModelForSequenceClassification
    model_path=Path(model_dir)
    if not model_path.exists():
        print(f"\n[ERROR] Model folder not found: {model_path.resolve()}\n"); sys.exit(1)
    print(f"\n{'─'*50}\n  Nepali Hate Speech Detection System\n  NepBERTa-V5 · KEC CT707\n{'─'*50}")
    print(f"\n  Loading model from: {model_path.resolve()}")
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer=AutoTokenizer.from_pretrained(str(model_path),use_fast=False)
    model=AutoModelForSequenceClassification.from_pretrained(str(model_path),num_labels=2)
    model.to(device);model.eval()
    print(f"  Model loaded on {device} ✓\n  Opening browser at http://localhost:{port}\n  Press Ctrl+C to stop\n{'─'*50}\n")
    template_dir=Path(__file__).parent/"templates"
    app=Flask(__name__,template_folder=str(template_dir))
    CORS(app)
    @app.route("/")
    def index(): return render_template("index.html")
    @app.route("/predict",methods=["POST"])
    def predict():
        payload=request.get_json(force=True,silent=True) or {}
        raw_text=(payload.get("text") or "").strip()
        if not raw_text: return jsonify({"error":"No text provided."}),400
        clean=_preprocess(raw_text)
        if len(clean)<=1: return jsonify({"error":"Text too short or contains no Devanagari characters."}),400
        enc=tokenizer(clean,max_length=128,padding="max_length",truncation=True,return_tensors="pt")
        input_ids=enc["input_ids"].to(device);attention_mask=enc["attention_mask"].to(device)
        with torch.no_grad():
            logits=model(input_ids=input_ids,attention_mask=attention_mask).logits
            probs=torch.softmax(logits,dim=-1).squeeze().tolist()
        pred_id=int(torch.argmax(logits,dim=-1).item());is_hate=pred_id==1
        confidence=round(probs[pred_id]*100,2)
        return jsonify({"original_text":raw_text,"cleaned_text":clean,"label":"HATE" if is_hate else "NON-HATE",
            "is_hate":is_hate,"confidence":confidence,"prob_hate":round(probs[1]*100,2),
            "prob_non_hate":round(probs[0]*100,2),"output_text":_mask(clean) if is_hate else clean})
    return app
def main():
    parser=argparse.ArgumentParser(description="Nepali Hate Speech Detection — opens browser UI")
    parser.add_argument("--model","-m",default="Model",help="Path to your NepBERTa model folder")
    parser.add_argument("--port","-p",type=int,default=5002,help="Port (default: 5002)")
    parser.add_argument("--no-browser",action="store_true",help="Don't auto-open browser")
    args=parser.parse_args()
    app=create_app(model_dir=args.model,port=args.port)
    if not args.no_browser:
        def open_browser():
            time.sleep(1.5); webbrowser.open(f"http://localhost:{args.port}")
        threading.Thread(target=open_browser,daemon=True).start()
    app.run(host="0.0.0.0",port=args.port,debug=False)
if __name__=="__main__":
    main()
