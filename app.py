from flask import Flask,render_template,request,jsonify
import os
from flask_cors import CORS,cross_origin

from cnnClassifier.utils.common import decodeImage
from cnnClassifier.pipeline.stage_05_prediction import PredictionPipeline


os.putenv('LANG','en_US.UTF-8')
os.putenv('LC_ALL','en_US.UTF-8')


app = Flask(__name__)

CORS(app=app)


class ClientApp:
    def __init__(self):
        self.filaname = "inputImage.jpg"
        self.classifier = PredictionPipeline(self.filaname)

clApp = ClientApp()


@app.route('/',methods= ['GET'])
@cross_origin()
def home():
    return render_template('index.html')


 

@app.route('/train',methods= ['GET','POST'])
@cross_origin()
def trainRoute():
    os.system("dvc repro")
    return "Training Done Successfully"




@app.route('/predict',methods= ['GET','POST'])
@cross_origin()
def predictROUTE():
    image = request.json['image']
    decodeImage(image,clApp.filaname)
    result = clApp.classifier.predict()
    return jsonify(result)



if __name__ == "__main__":
    clApp = ClientApp()
    app.run(host='0.0.0.0',port=8080,debug=True)
