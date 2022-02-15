from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def main():
        return render_template("main.html")
def lineman():
        return render_template("lineman.html")

if __name__ == '__main__':
    app.run(debug=True)