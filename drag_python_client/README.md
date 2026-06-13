### DragScores Python Client

A minimal Python client to interact with the `DragScores` smart contract and to create off-chain signatures that are compatible with the contract's verification logic (`toEthSignedMessageHash(bytes(message))` + `ECDSA.recover`).

### Install

- Create/activate your virtual environment (recommended).
- Install dependencies:

```bash
pip install -r requirements.txt
```

### Start local Hardhat network and deploy contract

In a separate terminal, from `Reliable-dRAG-anonymous/drag_contract`:

```bash
npm run node
```

In another terminal, deploy to localhost:

```bash
npm run deploy:local
```

This will create Ignition deployment files under `drag_contract/ignition/deployments/chain-31337/` that the client can auto-discover.

### Usage

Run the example after the local node is up and deployment is done (from this directory):

```bash
cd Reliable-dRAG-anonymous/drag_python_client
python -m drag_python_client.examples.test_local
```

You may also explicitly pass an address via env var (overrides auto-discovery):

```bash
export DRAG_SCORES_ADDRESS=0xYourDeployedAddress
python -m drag_python_client.examples.test_local
```

### Notes

- Off-chain signing uses `personal_sign` semantics via `eth_account.messages.encode_defunct(text=...)` which matches the contract's `toEthSignedMessageHash(bytes(message))` check.
- Example uses the first two Hardhat accounts for owner and a source address.


