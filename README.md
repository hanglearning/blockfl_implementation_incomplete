# BlockFL implementation
From Paper - https://arxiv.org/abs/1808.03949
ref - https://github.com/satwikkansal/python_blockchain_app/tree/ibm_blockchain_post
### Register a new node with an existing node
Example - register 5001 with an existing 5000 node.
```
curl -X POST \
  http://127.0.0.1:5001/register_with \
  -H 'Content-Type: application/json' \
  -d '{"register_with_node_address": "http://127.0.0.1:5000"}'
```

### Get chain
```
curl -X GET http://localhost:5000/chain
curl -X GET http://localhost:5001/chain
```

### Kill port
```
sudo lsof -i tcp:500
kill -9 <PID>  
```

## Debugging flow
@Rerun Miner
```
export FLASK_APP=node_server_miner.py
flask run --port 5000
```
@Rerun Worker
```
export FLASK_APP=node_server_worker.py
flask run --port 5001
```
@Register node
```
curl -X POST \
  http://127.0.0.1:5001/register_with \
  -H 'Content-Type: application/json' \
  -d '{"register_with_node_address": "http://127.0.0.1:5000"}'
```
@run Miner
```
curl -X GET http://localhost:5000
```
@run Worker
```
curl -X GET http://localhost:5001
```