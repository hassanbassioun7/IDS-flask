import numpy as np
import pandas as pd
import os
from flask import Flask, jsonify, render_template, request, redirect, flash, send_file, url_for, session, make_response
from datetime import timedelta
from flask_mysqldb import MySQL
import mysql.connector
from werkzeug.utils import secure_filename
import pickle
import urllib.request
from datetime import datetime
from keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.preprocessing import LabelEncoder
import time
import threading


basedir = os.path.abspath(os.path.dirname(__file__))

intrusion = pickle.load(open('models/H5_PKL/intrusion.pkl', 'rb'))
kddDnnModel = load_model('models/H5_PKL/kddDnnFinal.h5')

with open('models/H5_PKL/outcomesFinal.pkl', 'rb') as f:
    outcomes = pickle.load(f)

progress = 0

conn = mysql.connector.connect(host="localhost", user="root", password="", database="ids")
cursor = conn.cursor()

app = Flask(__name__)
app.secret_key = "secret key"
app.permanent_session_lifetime = timedelta(days=1)

app.config.update(
    UPLOADED_PATH = os.path.join(basedir, 'uploads'),
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024,
    ALLOWED_EXTENSIONS = set(['.csv, .xlsx'])
    )

MAX_FILE_SIZE = 25 * 1024 * 1024 



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
    

# ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
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
# ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    


@app.route('/uploadCsv', methods=['GET', 'POST'])
def uploadCsv():
    global progress
    if request.method == 'POST':
        progress = 0  # Reset progress at the start of each upload
        file = request.files['dataset']
        filename = secure_filename(file.filename)
        file_path = os.path.join('uploads', filename)
        file.save(file_path)

        file_size = os.path.getsize(file_path)

        # Validate file type (CSV)
        if not filename.lower().endswith('.csv'):
            return jsonify({'error': 'Please upload a CSV file.'}), 400
        
        elif file_size > MAX_FILE_SIZE:
            return jsonify({'error': 'File size exceeds the limit (25MB).'}), 400

        # Start the training in a new thread
        threading.Thread(target=train_model, args=(file_path,)).start()
        
        # You can return a message or redirect to another page
        return jsonify({'message': 'File uploaded and model training started'}), 202
    
    if 'loggedin' in session:
        return render_template('uploadCsv.html', username = session['username'])
    else:
        return render_template('login.html')


def train_model(file_path):
    global progress
    data = pd.read_csv(file_path)
    
    # Automatically detect and label encode categorical features
    categorical_cols = data.select_dtypes(include=['object']).columns
    le = LabelEncoder()
    for col in categorical_cols:
        data[col] = le.fit_transform(data[col])
    
    y = data.iloc[:, -1]
    X = data.iloc[:, :-1]
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    # Initialize the model
    model = SGDClassifier(loss='log')
    
    # Determine the total number of iterations (for simplicity, let's assume 10)
    total_iterations = 10
    for i in range(total_iterations):
        # Update the model with a partial fit
        model.fit(X_train, y_train)
        
        # Update progress
        progress = int((i + 1) / total_iterations * 100)
        
    # Save the trained model
    pickle.dump(model, open('model.pkl', 'wb'))
    
    # Ensure progress is set to 100 at the end
    progress = 100

@app.route('/progress')
def get_progress():
    return jsonify({'progress': progress})



@app.route('/kddPredictionDNN', methods = ['GET', 'POST'])
def kddPredictionDNN():
    if 'loggedin' in session:
        return render_template('kddPredictionDNN.html', username = session['username'])
    else:
        return render_template('login.html')
    
    

@app.route('/kddPredictionDT', methods = ['GET', 'POST'])
def kddPredictionDT():
    if 'loggedin' in session:
        return render_template('kddPredictionDT.html', username = session['username'])
    else:
        return render_template('login.html')
    


@app.route('/DNNpredict', methods=['POST'])
def DNNpredict():
    int_feature = [float(x) for x in request.form.values()]
  
    final_features = np.array([int_feature])
   
    result = kddDnnModel.predict(final_features)
    predicted_label = outcomes[np.argmax(result)]  # Convert prediction to original label

    if 'loggedin' in session:
        return render_template('kddPredictionDNN.html', prediction_text=predicted_label, username=session['username'])
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
        return render_template('kddPredictionDT.html', prediction_text=i, username=session['username'])
    else:
        return render_template('login.html')
    


@app.route("/logout")
def logout():
    session.pop('loggedin', None)
    session.pop('username', None)
    return redirect(url_for('index'))



if __name__ == "__main__":
    app.run(debug=True)