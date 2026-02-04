from flask import Flask, request, render_template_string
import joblib

app = Flask(__name__)

model = joblib.load("model/spam_model.pkl")
vectorizer = joblib.load("model/vectorizer.pkl")

HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>AI Spam Detector</title>

<style>
*{
    margin:0;
    padding:0;
    box-sizing:border-box;
    font-family: 'Segoe UI', sans-serif;
}

body{
    height:100vh;
    background: linear-gradient(135deg,#0f2027,#203a43,#2c5364);
    display:flex;
    align-items:center;
    justify-content:center;
    color:white;
}

.card{
    width:420px;
    padding:30px;
    background:rgba(255,255,255,0.1);
    backdrop-filter: blur(15px);
    border-radius:20px;
    box-shadow:0 25px 50px rgba(0,0,0,0.4);
    transform-style: preserve-3d;
    transition:0.5s;
}

.card:hover{
    transform: rotateY(10deg) rotateX(5deg);
}

h2{
    text-align:center;
    margin-bottom:20px;
    letter-spacing:1px;
}

textarea{
    width:100%;
    height:120px;
    border:none;
    border-radius:10px;
    padding:15px;
    font-size:14px;
    outline:none;
    resize:none;
}

button{
    width:100%;
    margin-top:15px;
    padding:12px;
    border:none;
    border-radius:10px;
    background: linear-gradient(135deg,#00c6ff,#0072ff);
    color:white;
    font-size:16px;
    cursor:pointer;
    transition:0.4s;
}

button:hover{
    transform: translateY(-3px);
    box-shadow:0 10px 20px rgba(0,0,0,0.4);
}

.result{
    margin-top:20px;
    text-align:center;
    font-size:20px;
    font-weight:bold;
}

.spam{
    color:#ff4b5c;
}

.ham{
    color:#4cd137;
}

.footer{
    text-align:center;
    font-size:12px;
    opacity:0.7;
    margin-top:15px;
}
</style>

<script>
function animateResult(){
    const res = document.getElementById("res");
    res.style.transform = "scale(1.2)";
    setTimeout(()=>{res.style.transform="scale(1)";},300);
}
</script>

</head>

<body>

<div class="card">
    <h2>📧 AI Spam Detector</h2>

    <form method="post">
        <textarea name="msg" placeholder="Paste your email or message here..." required></textarea>
        <button onclick="animateResult()">Check Message</button>
    </form>

    {% if result %}
        <div id="res" class="result {{ cls }}">
            {{ result }}
        </div>
    {% endif %}

    <div class="footer">
        Powered by Machine Learning & Flask
    </div>
</div>

</body>
</html>
"""

@app.route("/", methods=["GET","POST"])
def home():
    result=""
    cls=""
    if request.method=="POST":
        msg=request.form["msg"]
        vec=vectorizer.transform([msg])
        pred=model.predict(vec)[0]

        if pred==1:
            result="🚨 Spam Detected"
            cls="spam"
        else:
            result="✅ Not Spam"
            cls="ham"

    return render_template_string(HTML, result=result, cls=cls)

if __name__=="__main__":
    app.run(debug=True)
