import datetime
import json

import requests
from flask import render_template, redirect, request

from app import app

# The node with which our frontend interacts, there can be multiple such nodes as well.

# input = ("Which port to connect?")
CONNECTED_NODE_ADDRESS = "http://127.0.0.1:8000"

blocks_to_show = []


def fetch_blockchain():
    """
    Function to fetch the chain from a blockchain node, parse the
    data and store it locally for displaying.
    """
    get_chain_address = f"{CONNECTED_NODE_ADDRESS}/chain"
    response = requests.get(get_chain_address)
    if response.status_code == 200:
        chain_meta = json.loads(response.content)
        chain_content = []
        # TODO
        for block in chain_meta["chain"]:
            chain_content.append(block)

        global blocks_to_show
        blocks_to_show = sorted(chain_content, key=lambda block: block['timestamp'],reverse=True)

# TODO readable_time=timestamp_to_string?
@app.route('/')
def index():
    fetch_updates()
    # TODO display node id and role
    return render_template('index.html',
                           title='BlockFL',
                           blocks_to_show=blocks_to_show,
                           node_address=CONNECTED_NODE_ADDRESS,
                           readable_time=timestamp_to_string)

# TODO design buttons to refresh chain
# @app.route('/submit', methods=['POST'])
# def submit_textarea():
#     """
#     Endpoint to create a new transaction via our application.
#     """
#     post_content = request.form["content"]
#     author = request.form["author"]

#     post_object = {
#         'author': author,
#         'content': post_content,
#     }

#     # Submit a transaction
#     new_tx_address = "{}/new_transaction".format(CONNECTED_NODE_ADDRESS)

#     requests.post(new_tx_address,
#                   json=post_object,
#                   headers={'Content-type': 'application/json'})

#     return redirect('/')


def timestamp_to_string(epoch_time):
    return datetime.datetime.fromtimestamp(epoch_time).strftime('%H:%M')
