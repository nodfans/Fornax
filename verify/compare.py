import numpy as np
import os
import json

def generate_test_vectors(output_dir="./output"):
    print("[Verify] Generating test vectors...")
    ir_path = os.path.join(output_dir, "model_ir.json")
    with open(ir_path, "r") as f:
        ir = json.load(f)
    
    target_op = ir["target_op"]
    weight_key = target_op["weight_key"]
    weight_meta = ir["weight_metadata"][weight_key]
    
    # Load quantized weights
    bin_path = os.path.join(output_dir, weight_meta["path"])
    weights = np.fromfile(bin_path, dtype=np.int8).reshape(target_op["out_features"], target_op["in_features"])
    
    # 1. Generate random input vector (1 x InFeatures)
    # We use small values to avoid immediate overflow in M1 simple tests
    input_vec = np.random.randint(-15, 15, size=(target_op["in_features"],), dtype=np.int8)
    
    # 2. Calculate expected output (INT32 accumulation)
    # y = x * W^T
    expected_out = np.matmul(input_vec.astype(np.int32), weights.T.astype(np.int32))
    
    # 3. Save to HEX files for Verilog
    testvec_dir = os.path.join(output_dir, "testvectors")
    os.makedirs(testvec_dir, exist_ok=True)
    
    with open(os.path.join(testvec_dir, "input.hex"), "w") as f:
        for v in input_vec.view(np.uint8):
            f.write(f"{v:02x}\n")
            
    with open(os.path.join(testvec_dir, "expected.hex"), "w") as f:
        for v in expected_out.view(np.uint32):
            # Write 8-byte hex for 32-bit int
            f.write(f"{v:08x}\n")
            
    print(f"[Verify] Test vectors saved to {testvec_dir}/")
    return input_vec, expected_out

def check_results(output_dir="./output"):
    print("\n[Verify] Comparing Simulation Results...")
    testvec_dir = os.path.join(output_dir, "testvectors")
    
    expected_path = os.path.join(testvec_dir, "expected.hex")
    actual_path = os.path.join(testvec_dir, "actual.hex")
    
    if not os.path.exists(actual_path):
        print(f"❌ actual.hex not found at {actual_path}. Did you run the simulation?")
        return
    
    with open(expected_path, "r") as f:
        expected = [int(line.strip(), 16) for line in f if line.strip()]
        # Convert back to signed int32 if needed (hex is uint32 view)
        expected = [e if e < 0x80000000 else e - 0x100000000 for e in expected]
        
    with open(actual_path, "r") as f:
        actual = [int(line.strip(), 16) for line in f if line.strip()]
        actual = [a if a < 0x80000000 else a - 0x100000000 for a in actual]
        
    match_count = 0
    for i in range(min(len(expected), len(actual))):
        if expected[i] == actual[i]:
            match_count += 1
        else:
            print(f"  [MISMATCH] Index {i}: Expected {expected[i]}, Got {actual[i]}")
            
    if match_count == len(expected) and len(expected) == len(actual):
        print(f"✅ PASS: All {match_count} outputs match perfectly!")
    else:
        print(f"❌ FAIL: {match_count}/{len(expected)} matches.")

def verify_m1(output_dir="./output"):
    print("--- [Stage 4: Verify] ---")
    
    # Step 1: Check files
    ir_path = os.path.join(output_dir, "model_ir.json")
    if not os.path.exists(ir_path):
        print("❌ model_ir.json not found. Run convert.py first.")
        return

    # Step 2: Generate test vectors
    generate_test_vectors(output_dir)
    
    # Step 3: Check for simulation output
    check_results(output_dir)

if __name__ == "__main__":
    verify_m1()
