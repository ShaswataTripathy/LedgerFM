# In-browser demo source

This is the source of https://huggingface.co/spaces/tripathyShaswata/LedgerFM-demo:
- `tokenizer.js` is a JavaScript port of `ledgerfm/tokens.py`, tested to give identical tokens.
- The page uses ONNX Runtime Web.
- `export_onnx.py` builds the int8 models. They are hosted on the Space, not here.
