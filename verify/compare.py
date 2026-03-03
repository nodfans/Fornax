import numpy as np
import os
import json

def generate_test_vectors(output_dir="./output"):
    print("[Verify] Generating test vectors for M2 Block...")
    ir_path = os.path.join(output_dir, "model_ir.json")
    with open(ir_path, "r") as f:
        ir = json.load(f)
    
    # Generate random input (1 x DIM)
    # Typically LLM hidden size is ir['ops'][0]['dim']
    dim = ir["ops"][0].get("dim") or ir["ops"][0].get("in_features")
    input_vec = np.random.randint(-10, 10, size=(dim,), dtype=np.int8)
    
    current_val = input_vec.astype(np.float32)
    
    # Run through the IR chain in Python
    results = {"block_input": current_val}
    
    for i, op in enumerate(ir["ops"]):
        weight_meta = ir["weight_metadata"][op["weight_key"]]
        bin_path = os.path.join(output_dir, weight_meta["path"])
        weights = np.fromfile(bin_path, dtype=np.int8)
        
        # Get input for this op
        op_input = results[op["input"]]
        
        if op["type"] == "layernorm":
            # Verilog: prod[14:7] which is (prod >> 7) & 0xFF
            prod = op_input.astype(np.int32) * weights.astype(np.int32)
            res_bits = (prod >> 7).astype(np.int8)
            results[op["name"]] = res_bits.astype(np.float32)
        elif op["type"] == "linear":
            weights = weights.reshape(op["out_features"], op["in_features"])
            res = np.matmul(op_input, weights.T.astype(np.float32)).astype(np.int32)
            
            # If not the last op, use bit-slicing [15:8]
            if i < len(ir["ops"]) - 1:
                res_bits = (res >> 8).astype(np.int8)
                results[op["name"]] = res_bits.astype(np.float32)
            else:
                # Final output: keep full 32-bit precision
                results[op["name"]] = res

    # Final output is the result of the last op
    expected_out = results[ir["ops"][-1]["name"]]
    
    # Save files
    testvec_dir = os.path.join(output_dir, "testvectors")
    os.makedirs(testvec_dir, exist_ok=True)
    with open(os.path.join(testvec_dir, "input.hex"), "w") as f:
        for v in input_vec.view(np.uint8):
            f.write(f"{v:02x}\n")
    with open(os.path.join(testvec_dir, "expected.hex"), "w") as f:
        # Match the width of the last op in top.v (typically 32-bit linear if last is linear)
        # For simplify M2, let's assume we output the 32-bit value 
        for v in expected_out.astype(np.int32).view(np.uint32):
            f.write(f"{v:08X}\n")

    print(f"[Verify] Test vectors saved.")

def check_results(output_dir="./output"):
    print("\n[Verify] Comparing Simulation Results...")
    expected_path = os.path.join(output_dir, "testvectors/expected.hex")
    actual_path = os.path.join(output_dir, "testvectors/actual.hex")
    
    if not os.path.exists(actual_path):
        print(f"❌ FAIL: Simulation output {actual_path} not found.")
        return

    def hex_to_signed_int32(h):
        val = int(h, 16)
        if val & (1 << 31): # Check if the sign bit (31st bit for 32-bit) is set
            val -= (1 << 32) # Convert to negative
        return val

    expected = []
    with open(expected_path, "r") as f:
        for line in f:
            if line.strip():
                expected.append(hex_to_signed_int32(line.strip()))
    
    actual = []
    with open(actual_path, "r") as f:
        for line in f:
            if line.strip():
                actual.append(hex_to_signed_int32(line.strip()))

    matches = 0
    total = len(expected)
    
    # We only check up to the len of actual received
    check_len = min(len(expected), len(actual))
    
    for i in range(check_len):
        if expected[i] == actual[i]:
            matches += 1
        else:
            # Limit output to first few mismatches and last few
            if matches < 10 or i > total - 5: 
                print(f"  [MISMATCH] Index {i}: Expected {expected[i]}, Got {actual[i]}")

    if len(actual) != len(expected):
        print(f"⚠️ Warning: Result length mismatch. Expected {len(expected)}, Got {len(actual)}")

    if matches == total and total > 0:
        print(f"✅ PASS: All {matches} outputs match perfectly!")
    else:
        print(f"❌ FAIL: {matches}/{total} matches.")

def verify_m1(output_dir="./output"):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen", action="store_true", help="Force generate new test vectors")
    parser.add_argument("--check", action="store_true", help="Only check existing simulation results")
    args, _ = parser.parse_known_args()

    print("--- [Stage 4: Verify] ---")
    
    # Step 1: Check files
    ir_path = os.path.join(output_dir, "model_ir.json")
    if not os.path.exists(ir_path):
        print("❌ model_ir.json not found. Run convert.py first.")
        return

    testvec_dir = os.path.join(output_dir, "testvectors")
    input_hex = os.path.join(testvec_dir, "input.hex")

    if args.check:
        check_results(output_dir)
        return

    # Step 2: Generate test vectors if forced or missing
    if args.gen or not os.path.exists(input_hex):
        generate_test_vectors(output_dir)
    else:
        print("[Verify] Using existing test vectors.")
    
    # Step 3: Check for simulation output
    check_results(output_dir)

if __name__ == "__main__":
    verify_m1()
