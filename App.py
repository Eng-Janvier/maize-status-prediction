from flask import Flask,render_template,jsonify
from flask.globals import request
import os
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from skimage import io
import numpy as np
from PIL import Image


UPLOAD_FOLDER = 'static/images/'
app = Flask(__name__)
app.config['UPLOAD_FOLDER']= UPLOAD_FOLDER





@app.route('/')
def home():
    return render_template('predection.html')

@app.route('/predict',methods=['POST'])
def prediction():

    ext_list = ['jpg','png','jpeg']
    if request.method =='POST':
        if 'myimage' in request.files:
            file = request.files['myimage']
            filename = secure_filename(file.filename)
            if  not filename=='':

                file.save(os.path.join(app.config['UPLOAD_FOLDER'],filename))
                img = io.imread('static/images/'+filename)
                image=Image.fromarray(img)
                image=image.resize((150,150))
                image=np.asarray(image)
                image=np.expand_dims(image,axis=0)
                model =load_model('Edson_model.h5')
                pred = model.predict(image)
                label = ['UNHEALTY' if str(pred[0][0])=='0.0' else 'HEATH']
                label = 'MAIZE IS :' +label[0]
                result = {
                    'image':filename,
                    'message':label
                }
                return render_template('result.html',result=result)
                    

            else:
                return 'no file sent or invalid format'



if __name__=="__main__":
    app.run(debug=True)
