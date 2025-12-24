from flask import Flask, request, render_template, redirect, url_for
import os
from processor import run_full_analysis

app = Flask(__name__)
UPLOAD_DIR = "uploads"
REPORT_DIR = r"D:\Project\aistudytools\AI and Plagiarism Detect\reports"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

@app.route("/")
def home():
    return render_template("upload.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    text_input = request.form.get("text", "").strip()
    file = request.files.get("file", None)
    allow_web = ("allow_web" in request.form)
    deep_citation = ("deep_citation" in request.form)
    if file:
        filepath = os.path.join(UPLOAD_DIR, file.filename)
        file.save(filepath)
        report_file = run_full_analysis(filepath=filepath, pasted_text=None, allow_web = allow_web, deep_citation = deep_citation)

    elif text_input:
        report_file = run_full_analysis(filepath=None, pasted_text=text_input, allow_web = allow_web, deep_citation = deep_citation)

    else:
        return "No input provided."

    # open report in new tab
    return redirect("/report/" + report_file)

@app.route("/report/<name>")
def report(name):
    with open(os.path.join(REPORT_DIR, name), "r", encoding="utf-8") as f:
        content = f.read()
    return content

if __name__ == "__main__":
    app.run(debug=True, port=5000)
