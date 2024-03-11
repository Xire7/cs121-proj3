import os
import sys
sys.path.append("../backend/scripts")
import main


from flask import Flask, render_template

app = Flask(__name__)


@app.route('/')
def home():
    print("home")
    return render_template("index.html")

@app.route('/result/<results>')
def results(results=None):
    if results:
        urls = main.main(results)
        if len(urls) > 10:
             pages = [urls [i:i + 6] for i in range(0, len(urls), 6) ]
        elif len(urls) == 0:
            return render_template("index.html")
        else:
            pages = [urls]
        return render_template("results.html",pages=pages, query=results, length=len(pages))
    
    
@app.route('/result')
def empty():
    print("empty")
    return render_template("index.html",empty=True)
