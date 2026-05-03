from flask import Flask, render_template, request
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import re

app = Flask(__name__)

sentiment_analyzer = pipeline("sentiment-analysis", model="blanchefort/rubert-base-cased-sentiment")

tokenizer = AutoTokenizer.from_pretrained("sberbank-ai/rugpt3medium_based_on_gpt2")#"hello" -> [32, 35,1, 2]
model = AutoModelForCausalLM.from_pretrained("sberbank-ai/rugpt3medium_based_on_gpt2")


def extract_film_title(generated_text):
    if not generated_text:
        return ""
    
    m = re.search(r'""[^"]{2, 100}', generated_text)
    if m:
        return m.group(1).strip()
    
    return generated_text.strip().splitlines()[0][:80].strip()

def generate_recommendation(mood):
    prompt = (f"Посоветуй один популярный фильм для человека, у которого {mood} настроение. Назови только один фильм и кратко объясни почему")

    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        **inputs,
        max_length=70,
        do_sample=True,
        top_p=0.9,
        temperature=0.8
    )

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return text[len(prompt):].strip()



@app.route("/", methods=["GET", "POST"])
def index():

    recommendation = ""
    user_text = ""

    if request.method == "POST":
        user_text = request.form["message"]
        mood = ""
        result = sentiment_analyzer(user_text)[0]
        label = result["label"]

        if label == "POSITIVE":
            mood = "хорошее"
        elif label == "NEGATIVE":
            mood = "плохое"
        else:
            mood = "нейтральное"
        
        ai_text = extract_film_title(generate_recommendation(mood))
        recommendation = f"Настроение: {mood}. <br>Рекомендация: {ai_text}"


    return render_template("index.html", recommendation=recommendation, user_text=user_text)

@app.route("/submit", methods=["POST"])
def submit():
    user_message = request.form.get("message", "")

    if not user_message.strip():
        reply = "Ты ничего не ввел"
    else:
        reply = f"Я получил твой текст {user_message}!"
    
    return render_template("result.html", user_message=user_message, reply=reply)


if __name__ == "__main__":
    app.run(debug=True)