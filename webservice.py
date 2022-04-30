from lib2to3.pytree import convert
import flask
from flask import request
from convert_to_img import convert_to_img
from predict_image import predict_image
from create_model import classno_to_charname

app = flask.Flask(__name__)
app.config["DEBUG"] = True

@app.route('/', methods=['POST'])
def classify():
    request_data = request.get_json()
    data = [[[dict['x'], -dict['y']] for dict in stroke] for stroke in request_data["data"]]
    print(data)
    img = convert_to_img(data)
    model_filename = "models/model20220418-174527.h5"
    class_no = predict_image(img, model_filename)
    return classno_to_charname(class_no)

if __name__ == '__main__':
    app.run(debug=True, port=8080, host="0.0.0.0")