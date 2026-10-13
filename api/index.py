import os, pickle
from pathlib import Path
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
app=FastAPI(title="OIBSIP ML Lab",version="1.0")

iris_bundle=joblib.load(ROOT/"iris_flower_detection"/"model"/"iris_model.pkl")
iris_model=iris_bundle["model"]; iris_encoder=iris_bundle["label_encoder"]
spam_model=joblib.load(ROOT/"email_spam_detection"/"model"/"spam_model.pkl")
spam_vectorizer=joblib.load(ROOT/"email_spam_detection"/"model"/"vectorizer.pkl")
with open(ROOT/"Car-Price-Prediction"/"model"/"car_price_model.pkl","rb") as f:
    car_bundle=pickle.load(f)
car_model=car_bundle["model"]; car_features=car_bundle["features"]

class IrisIn(BaseModel):
    sepal_length:float=5.1
    sepal_width:float=3.5
    petal_length:float=1.4
    petal_width:float=0.2
class SpamIn(BaseModel):
    text:str
class CarIn(BaseModel):
    year:int=2015
    present_price:float=5.0
    kms_driven:float=50000
    fuel_type:str="Petrol"
    seller_type:str="Dealer"
    transmission:str="Manual"
    owner:int=0

@app.get("/health")
def health(): return {"status":"ok","models":["iris_svm","spam_naive_bayes","car_random_forest"]}

@app.post("/api/iris")
def iris(x:IrisIn):
    arr=np.array([[x.sepal_length,x.sepal_width,x.petal_length,x.petal_width]])
    pred=iris_model.predict(arr)
    label=str(iris_encoder.inverse_transform(pred)[0])
    return {"prediction":label}

@app.post("/api/spam")
def spam(x:SpamIn):
    vec=spam_vectorizer.transform([x.text])
    pred=int(spam_model.predict(vec)[0])
    return {"prediction":"Spam" if pred==1 else "Not Spam","class":pred}

@app.post("/api/car")
def car(x:CarIn):
    d={
      "Present_Price":x.present_price,
      "Driven_kms":x.kms_driven,
      "Owner":x.owner,
      "Car_Age":2026-x.year,
      "Fuel_Type_Diesel":1 if x.fuel_type=="Diesel" else 0,
      "Fuel_Type_Petrol":1 if x.fuel_type=="Petrol" else 0,
      "Selling_type_Individual":1 if x.seller_type=="Individual" else 0,
      "Transmission_Manual":1 if x.transmission=="Manual" else 0,
    }
    frame=pd.DataFrame([d]).reindex(columns=car_features,fill_value=0)
    pred=float(car_model.predict(frame)[0])
    return {"prediction_lakhs":round(max(0,pred),2)}

HTML=r'''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>OIBSIP ML Lab</title><style>
:root{--bg:#080b10;--p:#121821;--l:#293441;--t:#f4f7f9;--m:#8998a5;--c:#6fe2ff;--g:#82f1b4;--v:#ae91ff}*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at 80% 0,#1f2741,transparent 31%),var(--bg);color:var(--t);font-family:Inter,system-ui,sans-serif}.s{width:min(1160px,calc(100% - 28px));margin:auto}.n{height:76px;border-bottom:1px solid #202b36;display:flex;align-items:center;justify-content:space-between}.b{display:flex;gap:10px;align-items:center}.mk{width:40px;height:40px;border-radius:13px;background:var(--c);color:#071117;display:grid;place-items:center;font-weight:900}.b small,.b strong{display:block}.b small{font-size:10px;color:var(--m)}a{color:#b3bdc7;text-decoration:none;font-size:12px}.h{padding:60px 0 28px}.e{font-size:10px;color:var(--c);font-weight:850;text-transform:uppercase;letter-spacing:.18em}.h h1{font-size:clamp(44px,6.2vw,72px);line-height:1;letter-spacing:-.05em;margin:13px 0 17px}.h p{color:#a5b0ba;max-width:760px;line-height:1.7}.tabs{display:flex;gap:7px;margin-bottom:13px;flex-wrap:wrap}.tab{border:1px solid #31404e;background:#151d26;color:#c8d1d7;border-radius:999px;padding:9px 12px;cursor:pointer;font-weight:750}.tab.on{background:var(--c);color:#071117;border-color:transparent}.panel{border:1px solid var(--l);background:linear-gradient(180deg,#131a23,#0e1319);border-radius:21px;padding:20px;margin-bottom:70px}.view{display:none}.view.on{display:grid;grid-template-columns:1fr 1fr;gap:18px}.fields{display:grid;grid-template-columns:1fr 1fr;gap:10px}.f label{display:block;font-size:9px;color:#758591;margin-bottom:5px;text-transform:uppercase}.f input,.f select,.f textarea{width:100%;border:1px solid #303e4b;background:#090e13;color:white;border-radius:11px;padding:11px}.f textarea{min-height:180px;resize:vertical}.go{border:0;border-radius:11px;background:var(--g);color:#07120b;padding:12px 14px;font-weight:900;cursor:pointer;margin-top:12px}.result{border:1px solid #293846;background:#0a1015;border-radius:16px;padding:20px;display:grid;place-items:center;text-align:center;min-height:280px}.result span{font-size:10px;color:#74838f;text-transform:uppercase;letter-spacing:.12em}.result strong{font-size:38px;letter-spacing:-.04em;margin-top:8px}.result p{font-size:10px;color:#7e8c98;line-height:1.6;max-width:360px}.foot{height:74px;border-top:1px solid #202b36;color:#65737e;font-size:10px;display:flex;align-items:center;justify-content:space-between}@media(max-width:760px){.view.on{grid-template-columns:1fr}.fields{grid-template-columns:1fr}} </style></head><body><div class="s"><header class="n"><div class="b"><span class="mk">ML</span><span><strong>OIBSIP ML Lab</strong><small>Three deployed models</small></span></div><a href="https://github.com/Sankalp-gupta1/OIBSIP">GitHub ↗</a></header><main><section class="h"><span class="e">Oasis Infobyte projects</span><h1>Three models. One clean live lab.</h1><p>The deployed app loads the actual saved model files from this repository: Iris SVM, email spam classifier and car-price Random Forest.</p></section><div class="tabs"><button class="tab on" data-v="iris">Iris classifier</button><button class="tab" data-v="spam">Spam detector</button><button class="tab" data-v="car">Car price</button></div><section class="panel">
<div id="iris" class="view on"><div><div class="fields"><div class="f"><label>Sepal length</label><input id="sl" type="number" step=".1" value="5.1"></div><div class="f"><label>Sepal width</label><input id="sw" type="number" step=".1" value="3.5"></div><div class="f"><label>Petal length</label><input id="pl" type="number" step=".1" value="1.4"></div><div class="f"><label>Petal width</label><input id="pw" type="number" step=".1" value=".2"></div></div><button class="go" onclick="iris()">Predict species</button></div><div class="result"><div><span>SVM prediction</span><strong id="ir">—</strong><p>Uses the repository's saved SVM + label encoder bundle.</p></div></div></div>
<div id="spam" class="view"><div><div class="f"><label>Email / SMS text</label><textarea id="msg">Congratulations! You have won a free prize. Click now to claim.</textarea></div><button class="go" onclick="spam()">Check message</button></div><div class="result"><div><span>Naive Bayes prediction</span><strong id="sr">—</strong><p>Uses the saved vectorizer and spam model from the original project.</p></div></div></div>
<div id="car" class="view"><div><div class="fields"><div class="f"><label>Year</label><input id="year" type="number" value="2018"></div><div class="f"><label>Present price (lakhs)</label><input id="pp" type="number" step=".1" value="8"></div><div class="f"><label>KMs driven</label><input id="km" type="number" value="40000"></div><div class="f"><label>Owner count</label><select id="owner"><option>0</option><option>1</option><option>3</option></select></div><div class="f"><label>Fuel</label><select id="fuel"><option>Petrol</option><option>Diesel</option><option>CNG</option></select></div><div class="f"><label>Seller</label><select id="seller"><option>Dealer</option><option>Individual</option></select></div><div class="f"><label>Transmission</label><select id="trans"><option>Manual</option><option>Automatic</option></select></div></div><button class="go" onclick="car()">Estimate price</button></div><div class="result"><div><span>Random Forest estimate</span><strong id="cr">—</strong><p>Uses the saved Random Forest model and original feature order.</p></div></div></div>
</section></main><footer class="foot"><span>Built by Sankalp Gupta</span><span>FastAPI · scikit-learn · saved models · Vercel</span></footer></div><script>
const $=s=>document.querySelector(s);document.querySelectorAll(".tab").forEach(b=>b.onclick=()=>{document.querySelectorAll(".tab").forEach(x=>x.classList.remove("on"));document.querySelectorAll(".view").forEach(x=>x.classList.remove("on"));b.classList.add("on");$("#"+b.dataset.v).classList.add("on")});
async function post(url,data){const r=await fetch(url,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(data)});if(!r.ok)throw Error("Request failed");return r.json()}
async function iris(){try{const d=await post("/api/iris",{sepal_length:+$("#sl").value,sepal_width:+$("#sw").value,petal_length:+$("#pl").value,petal_width:+$("#pw").value});$("#ir").textContent=d.prediction}catch(e){$("#ir").textContent="Error"}}
async function spam(){try{const d=await post("/api/spam",{text:$("#msg").value});$("#sr").textContent=d.prediction}catch(e){$("#sr").textContent="Error"}}
async function car(){try{const d=await post("/api/car",{year:+$("#year").value,present_price:+$("#pp").value,kms_driven:+$("#km").value,fuel_type:$("#fuel").value,seller_type:$("#seller").value,transmission:$("#trans").value,owner:+$("#owner").value});$("#cr").textContent="₹ "+d.prediction_lakhs.toFixed(2)+" L"}catch(e){$("#cr").textContent="Error"}}
iris()</script></body></html>'''

@app.get("/",response_class=HTMLResponse)
def root(): return HTML
