from flask import Flask, render_template, request
import docx
import google.generativeai as genai
import os

app = Flask(__name__)

# Step 1: Load TAJA document content
def load_taja_data(doc_path):
    doc = docx.Document(doc_path)
    full_text = "\n".join([para.text for para in doc.paragraphs if para.text.strip() != ""])
    return full_text

TAJA_CONTENT = load_taja_data("taja_data/Manataja.docx")

# Step 2: Set up Gemini Pro with your API key
genai.configure(api_key="AIzaSyA1e76W8N-Wf7dUZ7CBSofW9mLqyj8cFYw")
model = genai.GenerativeModel("gemini-pro")

# Step 3: Define chatbot route
@app.route("/", methods=["GET", "POST"])
def index():
    response = ""
    if request.method == "POST":
        user_input = request.form["user_input"]
        prompt = f"You are TAJA's official AI assistant. Answer user questions using this content:\n\n{TAJA_CONTENT}\n\nUser: {user_input}\nTAJA Bot:"
        result = model.generate_content(prompt)
        response = result.text
    return render_template("index.html", response=response)

if __name__ == "__main__":
    app.run(debug=True)
