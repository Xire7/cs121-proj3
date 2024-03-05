import os
import sys
sys.path.append("../backend/scripts")
import main


from flask import Flask, render_template

app = Flask(__name__)


@app.route('/')
def run():
    urls = main.main()
    return render_template("index.html",urls=urls)
    
