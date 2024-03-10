import os
import sys
sys.path.append("../backend/scripts")
import main


from flask import Flask, render_template

app = Flask(__name__)


@app.route('/')
def home():
    print("home")
    return render_template("index.html", content=True)

@app.route('/result/<results>')
def results(results=None):
    if results:
        urls = main.main(results)
        print("main")
        if len(urls) > 15:
             pages = [urls [i:i + 6] for i in range(0, len(urls), 6) ]
        else:
            pages = urls
        print(pages)
        return render_template("results.html",pages=pages, query=results, length=len(pages))
    #empty or not found page later
    print("emtty")
    
    
@app.route('/result')
def empty():
    print("empty")
    return render_template("index.html")