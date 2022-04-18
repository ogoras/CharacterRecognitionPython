from lib2to3.pytree import convert
import flask
from flask import request
from convert_to_img import convert_to_img

app = flask.Flask(__name__)
app.config["DEBUG"] = True

@app.route('/', methods=['POST'])
def classify():
    request_data = request.get_json()
    data = [[[dict['x'], -dict['y']] for dict in stroke] for stroke in request_data["data"]]
    print(data)
    img = convert_to_img(data)
    img.show()
    return 'not implemented'

if __name__ == '__main__':
    app.run(debug=True, port=8000)