"""
Fornax — Stage 2: Converter
---------------------------
Quantize float32 weights → INT8 and build Model IR.

Usage:
    from src.converter import ModelConverter
    converter = ModelConverter("./output")
    converter.convert()
    converter.save("./output")
"""

import os
import json
import numpy as np


class ModelConverter:
    def __init__(self, data_dir: str):
        """
        Args:
            data_dir: Path to directory containing graph.json and weights/ folder.
        """
        self.data_dir = data_dir
        self.graph = {}
        self.weights = {}
        self.quantized_weights = {}  # { name: (int8_data, scale) }
        self.model_ir = {}

        self._load_data()

    def _load_data(self):
        """Load graph.json and available .npy weights."""
        graph_path = os.path.join(self.data_dir, "graph.json")
        with open(graph_path, "r") as f:
            self.graph = json.load(f)

        weights_dir = os.path.join(self.data_dir, "weights")
        if os.path.exists(weights_dir):
            for f in os.listdir(weights_dir):
                if f.endswith(".npy"):
                    name = f[:-4]  # name with underscores
                    self.weights[name] = np.load(os.path.join(weights_dir, f))

    def convert(self):
        """Perform symmetric INT8 quantization and build IR."""
        print("[Converter] Quantizing weights to INT8...")
        for name, weight in self.weights.items():
            self.quantized_weights[name] = self._quantize_symmetric(weight)

        print("[Converter] Building Model IR...")
        self._build_ir()
        return self

    def save(self, output_dir: str):
        """Save q-weights and model_ir.json."""
        q_weights_dir = os.path.join(output_dir, "quantized_weights")
        os.makedirs(q_weights_dir, exist_ok=True)

        # Save quantized weights (.bin) and metadata
        weight_metadata = {}
        for name, (q_data, scale) in self.quantized_weights.items():
            path = os.path.join(q_weights_dir, f"{name}.bin")
            q_data.tofile(path)
            weight_metadata[name] = {
                "path": f"quantized_weights/{name}.bin",
                "scale": float(scale),
                "shape": list(q_data.shape),
                "dtype": "int8"
            }

        # Update IR with weight metadata
        self.model_ir["weight_metadata"] = weight_metadata

        ir_path = os.path.join(output_dir, "model_ir.json")
        with open(ir_path, "w") as f:
            json.dump(self.model_ir, f, indent=2)

        print(f"[Converter] Saved {len(self.quantized_weights)} quantized tensors.")
        print(f"[Converter] IR saved to {ir_path}")

    def _quantize_symmetric(self, weight: np.ndarray):
        """
        Symmetric INT8 quantization.
        scale = max(abs(weight)) / 127
        q = round(weight / scale)
        """
        max_val = np.max(np.abs(weight))
        if max_val == 0:
            return np.zeros(weight.shape, dtype=np.int8), 1.0
        
        scale = max_val / 127.0
        q_weight = np.round(weight / scale).astype(np.int8)
        return q_weight, scale

    def _build_ir(self):
        """Construct the IR representing the hardware flow."""
        # For M1, we focus on the first linear layer found in the graph
        first_layer = self.graph["layers"][0]
        q_proj = next(op for op in first_layer["ops"] if op["name"] == "q_proj")

        self.model_ir = {
            "version": "M1",
            "model_id": self.graph["model_id"],
            "target_op": {
                "type": "linear",
                "name": q_proj["name"],
                "in_features": q_proj["in_dim"],
                "out_features": q_proj["out_dim"],
                "weight_key": q_proj["weight"]
            },
            "global_config": {
                "precision": "int8",
                "quantization": "symmetric"
            }
        }
