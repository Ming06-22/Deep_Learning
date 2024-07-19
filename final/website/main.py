from flask import Flask, url_for, redirect, render_template, request
from load_model import *
import time
app = Flask(__name__)

@app.route("/", methods = ["GET"])
def home():
    return render_template("index.html")

@app.route("/uplaod_file", methods = ["POST"])
def upload_file():
    file = request.files["filename"]
    filename = file.filename.split(".")[0]
    while filename[-1].isnumeric():
        filename = filename[: -1]
    file.save("./static/test/test/photo.png")
    
    result = load_model_predict()
    
    return render_template("result.html", result = result, label = filename)
    
if __name__ == "__main__":
    app.run()