# BlockFL implementation (INCOMPLETE, Please Read IMPORTANT)

From Paper - https://arxiv.org/abs/1808.03949 
Blockchain Structure Tutorial - https://github.com/satwikkansal/python_blockchain_app/tree/ibm_blockchain_post

### 🔴IMPORTANT❗🔴
This repo has been deprecated and is unlikely to receive further updates.

This repo hosts an incomplete and immature BlockFL implementation in Python, and the overall codebase may have limited research value. However, you may find it worthwhile to reuse some of the functions written in this repo and extend it in your own work. This repo also has an executable version of the code in the **dev branch**. If interested, please watch the following YouTube video to understand the execution steps and see which parts of the code you may wish to reuse
https://www.youtube.com/watch?v=rL1S6vQn_wM
### Envs used for this repo
```
python 3.7.7
Flask 1.1.1
```

## Sample Running Flow
Run a Miner on port 5000
```
export FLASK_APP=node_server_miner.py
flask run --port 5000
```
Run a Worker on port 5001
```
export FLASK_APP=node_server_worker.py
flask run --port 5001
```
Register one node with the other
```
curl -X POST \
  http://127.0.0.1:5001/register_with \
  -H 'Content-Type: application/json' \
  -d '{"registrar_node_address": "http://127.0.0.1:5000"}'
```
Run the Miner
```
curl -X GET http://localhost:5000
```
Run the Worker
```
curl -X GET http://localhost:5001
```
## Available End Points
### Get chain
```
curl -X GET http://localhost:5000/chain
```

### Get metadata of a node, such as chain length and peer list
```
curl -X GET http://localhost:5000/get_chain_meta
```

### Get peer list
```
curl -X GET http://localhost:5000/get_peers
```

### Get role
```
curl -X GET http://localhost:5000/get_role
```

### Get a miner's communication round (MUST request from a miner node)
```
curl -X GET http://localhost:5000/get_miner_comm_round
```
