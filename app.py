import numpy as np
import pandas as pd
import os
from flask import Flask, jsonify, render_template, request, redirect, flash, send_file, url_for, session, make_response
from datetime import timedelta
import mysql.connector
from werkzeug.utils import secure_filename
import pickle
import urllib.request
from datetime import datetime
from keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.preprocessing import LabelEncoder
from email.mime.text import MIMEText
import time
import threading
import smtplib


basedir = os.path.abspath(os.path.dirname(__file__))

intrusion = pickle.load(open('models/intrusion.pkl', 'rb'))
kddDnnModel = load_model('models/kddDnnFinal.h5')

with open('models/outcomesFinal.pkl', 'rb') as f:
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
    
@app.route('/about', methods=['GET', 'POST'])
def about():
        if 'username' in session:
         return render_template('about.html', username = session['username'])
        else:
            return render_template('about.html')
 # ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////       
@app.route('/contact', methods=['GET', 'POST'])
def contact():
        if 'username' in session:
         return render_template('contact.html', username = session['username'])
        else:
            return render_template('contact.html')
        
@app.route('/submit', methods=['POST'])
def submit_form():
    if request.method == 'POST':
        first_name = request.form['first_name']
        last_name = request.form['last_name']
        email = request.form['email']
        message = request.form['message']

        # Send email
        send_email(first_name, last_name, email, message)
        return 'Form submitted successfully!'

def send_email(first_name, last_name, email, message):
    sender_email = "yusuf.msalem@gmail.com"  # Change this to your email address
    receiver_email = "yusuf.msalem@gmial.com"  # Change this to recipient email address
    subject = "Contact Form Submission"
    body = f"First Name: {first_name}\nLast Name: {last_name}\nEmail: {email}\nMessage: {message}"

    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From'] = sender_email
    msg['To'] = receiver_email

    smtp_server = smtplib.SMTP('smtp.gmail.com', 587)
    smtp_server.starttls()
    smtp_server.login(sender_email, "jlog qusb dhlb ckfu")  # Change this to your email password
    smtp_server.sendmail(sender_email, receiver_email, msg.as_string())
    smtp_server.quit()
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


@app.route("/logout")
def logout():
    session.pop('loggedin', None)
    session.pop('username', None)
    return redirect(url_for('index'))



if __name__ == "__main__":
    app.run(debug=True)
