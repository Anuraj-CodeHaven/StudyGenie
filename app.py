from flask import Flask, request, jsonify, render_template
from transformers import pipeline

app = Flask(__name__)

# -------------------- LOAD FREE AI MODEL --------------------
print("Initializing StudyGenie AI...")
generator = pipeline(
    "text2text-generation",
    model="google/flan-t5-small"
)
print("StudyGenie AI Ready!")

# -------------------- HELPER FUNCTION --------------------
def ask_ai(prompt, max_len=200):
    try:
        result = generator(
            prompt,
            max_new_tokens=max_len,
            temperature=0.7,
            repetition_penalty=2.0,
            do_sample=True
        )
        return result[0]["generated_text"].strip()
    except Exception as e:
        return "Error: Unable to generate response."

# -------------------- ROUTES --------------------

@app.route("/")
def home():
    return render_template("index.html")

# 1️⃣ Topic Explanation
@app.route("/explain", methods=["POST"])
def explain():
    topic = request.json.get("topic", "").strip()
    if not topic:
        return jsonify({"error": "Topic cannot be empty"})
    prompt = f"Explain {topic} in simple student-friendly language with examples. Write at least 30 sentences."
    return jsonify({"result": ask_ai(prompt, 300)})

# 2️⃣ Flashcards (Q&A)
@app.route("/flashcard", methods=["POST"])
def flashcard():
    topic = request.json.get("topic", "").strip()
    if not topic:
        return jsonify({"error": "Topic cannot be empty"})
    q_prompt = f"Create one important question for revision on {topic}."
    a_prompt = f"Answer the question about {topic} in detail with at least 30 sentences."
    return jsonify({
        "question": ask_ai(q_prompt, 60),
        "answer": ask_ai(a_prompt, 300)
    })

# 3️⃣ Summarize Notes
@app.route("/summarize", methods=["POST"])
def summarize():
    notes = request.json.get("notes", "").strip()
    if not notes:
        return jsonify({"error": "Notes cannot be empty"})
    prompt = f"Summarize the following notes into clear bullet points with explanations:\n{notes}"
    return jsonify({"result": ask_ai(prompt, 250)})

# 4️⃣ Exam Tips & Key Points
@app.route("/exam_tips", methods=["POST"])
def exam_tips():
    topic = request.json.get("topic", "").strip()
    if not topic:
        return jsonify({"error": "Topic cannot be empty"})
    prompt = f"Generate exam tips, key points, important facts, and formulas for {topic}. Write at least 30 sentences."
    return jsonify({"result": ask_ai(prompt, 300)})

# 5️⃣ Quiz (MCQs)
@app.route("/quiz", methods=["POST"])
def quiz():
    topic = request.json.get("topic", "").strip()
    if not topic:
        return jsonify({"error": "Topic cannot be empty"})
    prompt = f"Generate 5 multiple choice questions on {topic} with 4 options and show the correct answer."
    return jsonify({"result": ask_ai(prompt, 300)})

# -------------------- RUN SERVER --------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8007)
