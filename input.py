import tensorflow as tf
import sys
import os
import json
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
from pymongo import MongoClient
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

UPLOAD_FOLDER = 'upload'
ALLOWED_EXTENSIONS = set(['jpg', 'jpeg', 'JPG', 'JPEG'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
client = MongoClient('localhost', 27017)
db = client.plant
collection = db.values

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            print 'no file'
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            print 'no filename'
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            #file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            #file.save(os.path.join('upload', secure_filename(file.filename)))
            #filepath = os.path.join(os.sep, UPLOAD_FOLDER, filename)
            file.save(secure_filename(file.filename))
            filepath = (secure_filename(file.filename))
            #filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            #filepath = (os.path.join('upload', secure_filename(file.filename)))
            return redirect(url_for('classificationProcess',
                                    filename = filename, filepath = filepath))
    return '''
    <!doctype html>
    <html>
    <head>
      <title>Input Image Page</title>
    </head>
    <body>
      <h1>Choose Image</h1>
      <form action="" method=post enctype=multipart/form-data>
        <p>
          <input type=file name=file>
          <input type=submit value=Upload>
        </p>
      </form>
    </body>
    </html>
    '''

@app.route('/classify/<filepath>')
def classificationProcess(filepath):
    image_path = filepath
    image_data = tf.gfile.FastGFile(image_path, 'rb').read()
    label_lines = [line.rstrip() for line in tf.gfile.GFile("tf_files/retrained_labels.txt")]
    with tf.gfile.FastGFile("tf_files/retrained_graph.pb", 'rb') as f:

        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')

    with tf.Session() as sess:

        softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
        predictions = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': image_data})
        top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
        arr = {"predictions":{}}
        arr1 = []
        res = ""
        for node_id in top_k:
            human_string = label_lines[node_id]
            score = predictions[0][node_id]
            arr['predictions'][human_string] = str(score)
            arr1.append(human_string)
        arr['result'] = arr1[0]
        confidence = int(round(float(arr["predictions"][arr1[0]])*100))
        arr['result_confidence'] = str(confidence) + "%"
        return redirect(url_for('showOutputPage', plant = arr1[0]))

@app.route('/Output/<plant>')
def showOutputPage(plant):
    return render_template('boutput.html', hPlantName = plant.title(), hBotanicalName = 'Botanical Name:', botanicalName = getBotanicalName(plant), hFamily = 'Family:', family = getFamily(plant), hAbout = 'About:', about = getAbout(plant), hMedicinalUses = 'Medicinal Uses:', medicinalUses = getMedicinalValues(plant))

@app.route('/getBotanicalName/<plant>')
def getBotanicalName(plant):
    try:
        for obj in collection.find({"plantName" : plant.title()}):
            return (obj['botanicalName'])

    except Exception, e:
        return str(e)

@app.route('/getFamily/<plant>')
def getFamily(plant):
    try:
        for obj in collection.find({"plantName" : plant.title()}):
            return (obj['family'])

    except Exception, e:
        return str(e)

@app.route('/getAbout/<plant>')
def getAbout(plant):
    try:
        for obj in collection.find({"plantName" : plant.title()}):
            return (obj['about'])

    except Exception, e:
        return str(e)

@app.route('/getMedicinalValues/<plant>')
def getMedicinalValues(plant):
    try:
        for obj in collection.find({"plantName" : plant.title()}):
            return (obj['medicinalUses'])

    except Exception, e:
        return str(e)


if __name__ == "__main__":
    app.run(debug=True)
