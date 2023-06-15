import json

from flask import Flask, request
import requests
from pmed import PMedianProblem


app = Flask(__name__)


@app.route('/')
def index():
    return 'Hello'


@app.route('/api/internal/create', methods=['POST'])
def optimize():
    # url = f"http://192.168.4.165:8000/api/tasks/{data['meta']['task_id']}"
    data = request.json

    task_id = data['meta']['task_id']

    pmed = PMedianProblem('ATM', data['p'], data['distances'], data['locations'])
    optimal_choices = pmed.solve()
    result = json.dumps(optimal_choices)
    return result


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
