Input:
- Option A: pre-calculated graph $G$
- Option B: node features V and positions X

Graph $G$ needs:
- node features V
- edge index E
- edge features e

Functionality:
- initialize with method and dimension
- train
- embed
optional:
- graph construction based on V, X

#### Notes
- make sure it is compatible with all systems including gpu supports with both cuda and MPS


#### Initialize
- define parameters
    - A) overwride defaults individually (e.g. set k=6)
    - B) input config as dictionary (config=config_dict)

optional:
- define method (regular, subgraph, nystrom)
alternatively:
- ignore subgraph, work only on given nodes (no superresolution) always assume nystrom approximation

#### Train
- implement with early stopping as default (but can be set to False)
- set default training parameters but they can be changed via argurments
- report loss
- return model

#### Embed
- simple forward pass that returns embeddings based on input. Also report loss.

- graph construction based on V, X
