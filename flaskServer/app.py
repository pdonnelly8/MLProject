import json
import predictions
from flask import Flask, request
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = 'files'


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/')
def root():
    return app.send_static_file('index.html')


@app.route('/audio', methods=['POST'])
def save_record():
    if request.method == 'POST':
      f = request.files['file']
      f.save(secure_filename(f.filename))
      print('D:\\pauricdonnellyfinalyearproject\\flaskServer\\' + f.filename)
      prediction = predictions.predict_classification('D:\\pauricdonnellyfinalyearproject\\flaskServer\\' + f.filename)
      if(prediction == 0):
        result = "Person does not have stroke"
      else:
        result = "Person may have stroke, please seek medical attention"  
    return json.dumps({ "text": result }), 200