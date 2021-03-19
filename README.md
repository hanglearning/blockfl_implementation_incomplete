# BlockFL implementation

From Paper - https://arxiv.org/abs/1808.03949 
Blockchain Structure Tutorial - https://github.com/satwikkansal/python_blockchain_app/tree/ibm_blockchain_post

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
