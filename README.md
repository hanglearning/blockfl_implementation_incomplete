# thesis_project
senior thesis
ref - https://github.com/satwikkansal/python_blockchain_app/tree/ibm_blockchain_post
### Register a new node with an existing node
Example - register 5001 with an existing 5000 node.
```
curl -X POST \
  http://127.0.0.1:5001/register_with \
  -H 'Content-Type: application/json' \
  -d '{"register_with_node_address": "http://127.0.0.1:5000"}'
```

