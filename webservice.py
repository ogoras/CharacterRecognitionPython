import flask
from flask import request
import convert_to_img

app = flask.Flask(__name__)
app.config["DEBUG"] = True

@app.route('/', methods=['POST'])
def classify():
    request_data = request.get_json()
    print(request_data)

    return 'not implemented'

if __name__ == '__main__':
    app.run(debug=True, port=8000)