from flask import Flask, render_template, jsonify, request, json
from predict import NB_predict,CNN_predict,LSTM_predict

app = Flask(__name__)


@app.route('/')
def trial():
    return render_template("index.html")


@app.route('/nb_predict', methods=['GET', 'POST'])
def naive_bayes():
    if request.method == 'POST':
        data = request.form['content']
        result = NB_predict(data)
        print(result[0])
        #print(type(result[0]))
        return jsonify(status=200, result=result[0])


@app.route('/cnn_predict', methods=['GET', 'POST'])
def convolution_nn():
    if request.method == 'POST':
        data = request.form['content']
        result = CNN_predict(data)
        return jsonify(status=200, result=result)


@app.route('/lstm_predict', methods=['GET', 'POST'])
def long_short_term_nn():
    if request.method == 'POST':
        data = request.form['content']
        result = LSTM_predict(data)
        return jsonify(status=200, result=result)


if __name__ == '__main__':
    app.run()
