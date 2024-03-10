from flask import Flask, render_template, request

app = Flask(__name__)

# Dummy function to simulate fetching search results
def get_search_results(query):
    # Replace this with your actual search logic
    return [
        ["Title 1", "Description 1", "example1.com"],
        ["Title 2", "Description 2", "example2.com"],
        ["Title 3", "Description 3", "example3.com"],
    ]

@app.route('/', methods=['GET', 'POST'])
def search():
    query = request.args.get('query', '')
    results = get_search_results(query) if query else []
    return render_template('search.html', query=query, pages=[results])

if __name__ == '__main__':
    app.run(debug=True)