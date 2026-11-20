"""Export LedgerFM encoder (v0.1) and Sparkov scorer to ONNX, then int8-quantise. Single-sequence inputs:
ints [1, L, 14] int64, floats [1, L, 12] float32 (one account window, no padding)."""
import json, sys, pathlib, numpy as np, torch, torch.nn as nn
from safetensors.torch import load_file
from ledgerfm.model import LedgerFM, Config
from onnxruntime.quantization import quantize_dynamic, QuantType

OUT = pathlib.Path("D:/workspace/projects/models/LedgerFM-static/models")
HERE = pathlib.Path(__file__).parent


class Wrap(nn.Module):
    def __init__(self, fm, head=None):
        super().__init__(); self.fm, self.head = fm, head

    def forward(self, ints, floats):
        L = ints.shape[1]
        seg = torch.ones_like(ints[..., 0]); pos = torch.arange(L, device=ints.device).unsqueeze(0)
        h = self.fm.encode(ints, floats, seg, pos)
        return torch.sigmoid(self.head(h)).squeeze(-1) if self.head is not None else h


def build():
    cfg = json.load(open("D:/workspace/projects/models/LedgerFM-release/weights/config.json"))["cfg"]
    enc = LedgerFM(Config(**cfg)); enc.load_state_dict(load_file("D:/workspace/projects/models/LedgerFM-release/weights/model.safetensors"))
    sd = load_file(HERE / "scorer/model.safetensors")
    sfm = LedgerFM(Config(**cfg)); sfm.load_state_dict({k[3:]: v for k, v in sd.items() if k.startswith("fm.")})
    head = nn.Sequential(nn.Linear(256, 256), nn.GELU(), nn.Dropout(0.1), nn.Linear(256, 1))
    head.load_state_dict({k[5:]: v for k, v in sd.items() if k.startswith("head.")})
    return Wrap(enc).eval(), Wrap(sfm, head).eval()


if __name__ == "__main__":
    enc, sc = build()
    ex = (torch.randint(1, 20, (1, 64, 14)), torch.randn(1, 64, 12))
    L = torch.export.Dim("L", min=1, max=128)
    for name, m, outn in [("encoder", enc, "emb"), ("scorer", sc, "prob")]:
        f = OUT / f"{name}_fp32.onnx"
        torch.onnx.export(m, ex, f, input_names=["ints", "floats"], output_names=[outn],
                          dynamic_shapes={"ints": {1: L}, "floats": {1: L}}, dynamo=True, external_data=False)
        quantize_dynamic(str(f), str(OUT / f"{name}.onnx"), weight_type=QuantType.QInt8,
                         op_types_to_quantize=["MatMul", "Gather"])
        print(name, f.stat().st_size // 2**20, "MB ->", (OUT / f"{name}.onnx").stat().st_size // 2**20, "MB")
