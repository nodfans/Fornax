"""
Fornax — Stage 1: Parser
------------------------
Load a HuggingFace model, extract weights and compute graph.

Usage:
    from src.parser import ModelParser

    parser = ModelParser("Qwen/Qwen2-0.5B")
    parser.parse()
    parser.save("./output")
"""

import os
import json
import numpy as np
from transformers import AutoModelForCausalLM, AutoConfig


class ModelParser:

    def __init__(self, model_id: str):
        """
        Args:
            model_id: HuggingFace model ID e.g. "Qwen/Qwen2-0.5B"
        """
        self.model_id = model_id
        self.model = None
        self.config = None
        self.weights = {}   # { layer_name: np.ndarray }
        self.graph = {}     # model structure metadata

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def parse(self):
        """Main entry point. Load model, extract weights + graph."""
        print(f"[Parser] Loading model: {self.model_id}")
        self._load_model()

        print(f"[Parser] Extracting weights...")
        self._extract_weights()

        print(f"[Parser] Building compute graph...")
        self._build_graph()

        print(f"[Parser] Done. {len(self.weights)} weight tensors extracted.")
        return self

    def save(self, output_dir: str):
        """Save weights as .npy files and graph as graph.json."""
        weights_dir = os.path.join(output_dir, "weights")
        os.makedirs(weights_dir, exist_ok=True)

        # Save each weight tensor
        for name, tensor in self.weights.items():
            safe_name = name.replace(".", "_")
            path = os.path.join(weights_dir, f"{safe_name}.npy")
            np.save(path, tensor)

        # Save compute graph
        graph_path = os.path.join(output_dir, "graph.json")
        with open(graph_path, "w") as f:
            json.dump(self.graph, f, indent=2)

        print(f"[Parser] Saved weights to {weights_dir}/")
        print(f"[Parser] Saved graph to {graph_path}")

    def get_single_layer(self, layer_idx: int = 0, proj: str = "q_proj"):
        """
        Helper for M1: extract just one Linear layer weight.

        Args:
            layer_idx: which transformer layer (0 = first)
            proj:      which projection (q_proj, k_proj, v_proj, o_proj)

        Returns:
            weight: np.ndarray of shape (out_features, in_features)
            name:   string name of the layer
        """
        key = f"model_layers_{layer_idx}_self_attn_{proj}_weight"
        if key not in self.weights:
            available = [k for k in self.weights if f"layers_{layer_idx}" in k]
            raise KeyError(
                f"Key '{key}' not found.\n"
                f"Available keys for layer {layer_idx}:\n"
                + "\n".join(f"  {k}" for k in available)
            )
        return self.weights[key], key

    # ------------------------------------------------------------------
    # Internal methods
    # ------------------------------------------------------------------

    def _load_model(self):
        """Load model from HuggingFace (downloads on first run, cached after)."""
        self.config = AutoConfig.from_pretrained(self.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype="auto",   # use model's native dtype
            device_map="cpu",     # keep on CPU, we just need weights
        )
        self.model.eval()

    def _extract_weights(self):
        """
        Extract all weight tensors from the model.
        Converts to float32 numpy arrays.
        Keys use underscores instead of dots for safe filenames.
        e.g. "model.layers.0.self_attn.q_proj.weight"
             → "model_layers_0_self_attn_q_proj_weight"
        """
        state_dict = self.model.state_dict()
        for name, tensor in state_dict.items():
            safe_name = name.replace(".", "_")
            self.weights[safe_name] = tensor.float().numpy()

    def _build_graph(self):
        """
        Build a simple metadata dict describing the model structure.
        This becomes graph.json — used by Stage 2 (converter).
        """
        cfg = self.config

        # Detect FFN activation type
        ffn_type = getattr(cfg, "hidden_act", "unknown")

        # Detect attention type
        num_kv_heads = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
        is_gqa = num_kv_heads != cfg.num_attention_heads  # grouped query attention

        self.graph = {
            "model_id":        self.model_id,
            "model_type":      cfg.model_type,
            "num_layers":      cfg.num_hidden_layers,
            "hidden_size":     cfg.hidden_size,
            "num_heads":       cfg.num_attention_heads,
            "num_kv_heads":    num_kv_heads,
            "is_gqa":          is_gqa,
            "head_dim":        cfg.hidden_size // cfg.num_attention_heads,
            "intermediate_size": getattr(cfg, "intermediate_size", None),
            "ffn_type":        ffn_type,
            "vocab_size":      cfg.vocab_size,
            "max_position_embeddings": getattr(cfg, "max_position_embeddings", None),
            "rope_theta":      getattr(cfg, "rope_theta", 10000.0),

            # Operator inventory — what Fornax needs to support
            "ops": [
                "embedding",
                "linear",       # q/k/v/o projections + FFN
                "layernorm",    # RMSNorm in most modern models
                "rope",         # rotary position embedding
                "softmax",
                "activation",   # SiLU / GELU etc.
            ],

            # Weight inventory — one entry per layer
            "layers": self._describe_layers(),
        }

    def _describe_layers(self):
        """Build a list of layer descriptors for graph.json."""
        layers = []
        cfg = self.config
        head_dim = cfg.hidden_size // cfg.num_attention_heads
        num_kv_heads = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)

        for i in range(cfg.num_hidden_layers):
            layers.append({
                "layer_idx": i,
                "ops": [
                    {
                        "type":    "linear",
                        "name":    "q_proj",
                        "weight":  f"model_layers_{i}_self_attn_q_proj_weight",
                        "in_dim":  cfg.hidden_size,
                        "out_dim": cfg.num_attention_heads * head_dim,
                    },
                    {
                        "type":    "linear",
                        "name":    "k_proj",
                        "weight":  f"model_layers_{i}_self_attn_k_proj_weight",
                        "in_dim":  cfg.hidden_size,
                        "out_dim": num_kv_heads * head_dim,
                    },
                    {
                        "type":    "linear",
                        "name":    "v_proj",
                        "weight":  f"model_layers_{i}_self_attn_v_proj_weight",
                        "in_dim":  cfg.hidden_size,
                        "out_dim": num_kv_heads * head_dim,
                    },
                    {
                        "type":    "linear",
                        "name":    "o_proj",
                        "weight":  f"model_layers_{i}_self_attn_o_proj_weight",
                        "in_dim":  cfg.hidden_size,
                        "out_dim": cfg.hidden_size,
                    },
                    {
                        "type":    "layernorm",
                        "name":    "input_layernorm",
                        "weight":  f"model_layers_{i}_input_layernorm_weight",
                    },
                    {
                        "type":    "layernorm",
                        "name":    "post_attention_layernorm",
                        "weight":  f"model_layers_{i}_post_attention_layernorm_weight",
                    },
                    {
                        "type":    "linear",
                        "name":    "gate_proj",
                        "weight":  f"model_layers_{i}_mlp_gate_proj_weight",
                        "in_dim":  cfg.hidden_size,
                        "out_dim": getattr(cfg, "intermediate_size", None),
                    },
                    {
                        "type":    "linear",
                        "name":    "up_proj",
                        "weight":  f"model_layers_{i}_mlp_up_proj_weight",
                        "in_dim":  cfg.hidden_size,
                        "out_dim": getattr(cfg, "intermediate_size", None),
                    },
                    {
                        "type":    "linear",
                        "name":    "down_proj",
                        "weight":  f"model_layers_{i}_mlp_down_proj_weight",
                        "in_dim":  getattr(cfg, "intermediate_size", None),
                        "out_dim": cfg.hidden_size,
                    },
                ],
            })
        return layers


# ------------------------------------------------------------------
# Quick test — run directly to verify parser works
# ------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    model_id = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2-0.5B"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "./output"

    parser = ModelParser(model_id)
    parser.parse()
    parser.save(output_dir)

    # M1 demo: print shape of q_proj from layer 0
    weight, name = parser.get_single_layer(layer_idx=0, proj="q_proj")
    print(f"\n[M1 Demo] Extracted: {name}")
    print(f"          Shape: {weight.shape}")
    print(f"          dtype: {weight.dtype}")
    print(f"          min={weight.min():.4f}  max={weight.max():.4f}")