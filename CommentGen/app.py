
from flask import Flask, request, jsonify
from comment import generate_comments
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/generate_comments', methods=['POST'])
def generate_comments_route():
    data = request.json
    print(f"Received data: {data}") 
    text = data.get('text')
    comments = generate_comments(text)
    print(f"Generated comments: {comments}")
    return jsonify(comments)

if __name__ == '__main__':
    app.run(debug=True)
