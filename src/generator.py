"""
Fornax — Stage 3: Generator
---------------------------
Emit Verilog code and weight hex files from Model IR.

Usage:
    from src.generator import VerilogGenerator
    gen = VerilogGenerator("./output")
    gen.generate()
"""

import os
import json
import numpy as np
from jinja2 import Environment, FileSystemLoader


class VerilogGenerator:
    def __init__(self, data_dir: str, template_dir: str = "./templates"):
        """
        Args:
            data_dir: Directory containing model_ir.json and quantized_weights/
            template_dir: Directory containing Jinja2 Verilog templates
        """
        self.data_dir = data_dir
        self.template_dir = template_dir
        self.output_dir = os.path.join(data_dir, "rtl")
        self.weights_out_dir = os.path.join(data_dir, "weights_hex")
        self.model_ir = {}

        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.weights_out_dir, exist_ok=True)

        self._load_ir()
        self.jinja_env = Environment(loader=FileSystemLoader(self.template_dir))

    def _load_ir(self):
        ir_path = os.path.join(self.data_dir, "model_ir.json")
        with open(ir_path, "r") as f:
            self.model_ir = json.load(f)

    def generate(self):
        """Generate Verilog modules and weight hex files."""
        print("[Generator] Generating Verilog for M1...")
        
        # 1. Generate weight HEX files (for $readmemh)
        self._generate_weight_hex()

        # 2. Render MatMul template
        self._render_matmul()

        print(f"[Generator] Verilog emitted to {self.output_dir}/")
        print(f"[Generator] Hex weights emitted to {self.weights_out_dir}/")

    def _generate_weight_hex(self):
        """Convert binary quantized weights to HEX string format for Verilog."""
        weight_metadata = self.model_ir.get("weight_metadata", {})
        
        for name, meta in weight_metadata.items():
            bin_path = os.path.join(self.data_dir, meta["path"])
            # Load as int8, view as uint8 for easy hex conversion
            weights = np.fromfile(bin_path, dtype=np.int8).view(np.uint8)
            
            # Convert to HEX (2 chars per byte)
            hex_path = os.path.join(self.weights_out_dir, f"{name}.hex")
            with open(hex_path, "w") as f:
                for w in weights:
                    f.write(f"{w:02x}\n")
            
            # Update meta with hex relative path for template
            meta["hex_path"] = f"../weights_hex/{name}.hex"

    def _render_matmul(self):
        """Render the matmul.v template using IR parameters."""
        target_op = self.model_ir["target_op"]
        weight_key = target_op["weight_key"]
        weight_meta = self.model_ir["weight_metadata"][weight_key]

        template = self.jinja_env.get_template("matmul.v.j2")
        rendered = template.render(
            in_features=target_op["in_features"],
            out_features=target_op["out_features"],
            weight_file=weight_meta["hex_path"]
        )

        with open(os.path.join(self.output_dir, "matmul.v"), "w") as f:
            f.write(rendered)
