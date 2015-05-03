from flask import Flask, request, jsonify
from flask import render_template
from tweeting import tweeting
from time import strptime, mktime
from datetime import datetime

app = Flask(__name__)

#homepage
@app.route('/')
def index():
	return render_template("index.html")
@app.route('/query')
def query():
	print request.args
	epicenter = [float(request.args.get('epiLt')), float(request.args.get('epiLn'))]
	print epicenter
	r1 = float(request.args.get('r1'))
	print r1
	r2 = float(request.args.get('r2'))
	print r2
	tI = request.args.get('start')
	tF = request.args.get('end')
	tI = datetime.fromtimestamp(mktime(strptime(tI, "%Y-%m-%d")))
	tF = datetime.fromtimestamp(mktime(strptime(tF, "%Y-%m-%d")))
	print tI
	print tF
	u = float(request.args.get('u'))
	res = tweeting(epicenter,r1,r2,tI,tF,u)
	return jsonify(res)

if __name__ == '__main__':
	app.run(host='127.0.0.1', port=3000, debug=True)
