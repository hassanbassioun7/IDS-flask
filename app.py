import numpy as np
import pandas as pd
import os
from flask import Flask, jsonify, render_template, request, redirect, flash, send_file, url_for, session, make_response
from flask_dropzone import Dropzone
from datetime import timedelta
from flask_mysqldb import MySQL
import mysql.connector
from werkzeug.utils import secure_filename
import pickle
import urllib.request
from datetime import datetime
from keras.models import load_model

basedir = os.path.abspath(os.path.dirname(__file__))

intrusion = pickle.load(open('model/intrusion.pkl', 'rb'))
kddDnnModel = load_model('model/kddDnnFinal.h5')

conn = mysql.connector.connect(host="localhost", user="root", password="", database="ids")
cursor = conn.cursor()

app = Flask(__name__)
app.secret_key = "secret key"
app.permanent_session_lifetime = timedelta(days=1)

app.config.update(
    UPLOADED_PATH = os.path.join(basedir, 'uploads'),
    DROPZONE_MAX_FILE_SIZE = 50,
    DROPZONE_TIMEOUT = 5 * 60 * 1000,
    DROPZONE_ALLOWED_FILE_CUSTOM = True,
    DROPZONE_ALLOWED_FILE_TYPE = '.csv, .xlsx',
    DROPZONE_INVALID_FILE_TYPE = "You can't upload files of this type. Upload CSV or XLSX only",
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024,
    ALLOWED_EXTENSIONS = set(['.csv, .xlsx'])
    )

@app.route("/")
def index():
    if 'username' in session:
        return render_template('index.html', username = session['username'])
    else:
        return render_template('index.html')
    
@app.route("/login", methods=['GET', 'POST'])
def login():
    msg=''
    if request.method=='POST':
        conn = mysql.connector.connect(host="localhost", user="root", password="", database="ids")
        cursor = conn.cursor()
        session.permanent = True
        username = request.form['username']
        password = request.form['password']
        cursor.execute('SELECT * FROM tbl_users WHERE username=%s AND password=%s', (username, password))
        record = cursor.fetchone()
        conn.close()
        cursor.close()
        if record:
            session['loggedin'] = True
            session['username'] = record[1]
            return redirect(url_for('index'))
        else:
            msg = 'Incorrect username or password. Try Again!'
    return render_template('login.html', msg=msg)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route("/select_dataset")
def select_dataset():
    if 'username' in session:
        return render_template('select_dataset.html', username = session['username'])
    else:
        return render_template('login.html')

@app.route("/upload_dataset", methods=['GET', 'POST'])
def upload_dataset():
    if request.method == "POST":
        filesize = request.cookies.get("filesize")
        file = request.files["file"]
        filename = secure_filename(file.filename)

        print(f"Filesize: {filesize}")
        print(file)

        if file.filename != "":
            filename = os.path.join(app.config['UPLOADED_PATH'], filename)
            file.save(filename)
        res = make_response(jsonify({"message": f"{file.filename} uploaded"}), 200)

        return res
    if 'loggedin' in session:
        return render_template('upload_dataset.html', username = session['username'])
    else:
        return render_template('login.html')

@app.route("/upload", methods=['GET', 'POST'])
def upload():
    if request.method == "POST":
        filesize = request.cookies.get("filesize")
        file = request.files["file"]

        print(f"Filesize: {filesize}")
        print(file)

        res = make_response(jsonify({"message": f"{file.filename} uploaded"}), 200)

        return res

    return render_template('upload.html')

dropzone = Dropzone(app)

@app.route('/dropzone', methods=['POST', 'GET'])
def dropzone():
    if request.method == 'POST':
        f = request.files.get('file')
        f.save(os.path.join(app.config['UPLOADED_PATH'], f.filename))
    return render_template('dropzone.html')

@app.route('/uploadX')
def uploadX():
    return render_template('uploadX.html')  

@app.route('/preview')
def previewX():
    return render_template('preview.html')  

@app.route('/previewX',methods=["POST"])
def preview():
    if request.method == 'POST':
        dataset = request.files['datasetfile']
        df = pd.read_csv(dataset,encoding = 'unicode_escape')
        df.set_index('Id', inplace=True)
    return render_template("previewX.html",df_view = df)

@app.route('/prediction', methods = ['GET', 'POST'])
def prediction():
    if 'loggedin' in session:
        return render_template('prediction.html', username = session['username'])
    else:
        return render_template('login.html')
    
@app.route('/predictionCopy', methods = ['GET', 'POST'])
def predictionCopy():
    if 'loggedin' in session:
        return render_template('prediction copy.html', username = session['username'])
    else:
        return render_template('login.html')

@app.route('/DTpredict', methods=['POST'])
def DTpredict():
    int_feature = [x for x in request.form.values()]
    
    final_features = [np.array(int_feature)]
    
    result = intrusion.predict(final_features)
    for i in result:
        print(i, end="")
     
    if 'loggedin' in session:
        return render_template('prediction.html', prediction_text=i, username=session['username'])
    else:
        return render_template('login.html')
    

with open('model/outcomesFinal.pkl', 'rb') as f:
    outcomes = pickle.load(f)
    
@app.route('/KDDpredict', methods=['POST'])
def KDDpredict():
    int_feature = [float(x) for x in request.form.values()]
  
    final_features = np.array([int_feature])
   
    result = kddDnnModel.predict(final_features)
    predicted_label = outcomes[np.argmax(result)]  # Convert prediction to original label

    if 'loggedin' in session:
        return render_template('prediction.html', prediction_text=predicted_label, username=session['username'])
    else:
        return render_template('login.html')


# -------------------------------------------

@app.route('/chart')
def chart():
	return render_template('chart.html')

@app.route('/performance')
def performance():
	return render_template('performance.html')   

@app.route("/registration")
def reg():
    return "Registration details"


# -----------------------------------------------

@app.route("/logout")
def logout():
    session.pop('loggedin', None)
    session.pop('username', None)
    return redirect(url_for('index'))

if __name__ == "__main__":
    app.run(debug=True)