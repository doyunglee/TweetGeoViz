from flask import Flask
app = Flask(__name__)

#homepage
@app.route('/')
def index():
	return flask.render_template("index.html")

if __name__ == '__main__':
    app.run()
