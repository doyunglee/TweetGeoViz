from flask import Flask
from flask import render_template
from tweeting import tweeting

app = Flask(__name__)

#homepage
@app.route('/')
def index():
	return flask.render_template("index.html")
@app.route('/query')
def query():
	epicenter = request.args.get('epi');
	r1 = request.args.get('r1');
	r2 = request.args.get('r2');
	tI = request.args.get('tI');
	tF = request.args.get('tF');
	res = tweeting(epicenter,r1,r2,tI,tF);
	return flask.jsonify(res);

if __name__ == '__main__':
	app.run(host='0.0.0.0', port=5000, debug=True)
