import argparse
import sys

def main():
    parser = argparse.ArgumentParser(description="Fornax: LLM to Verilog Compiler")
    parser.add_argument("--model", type=str, help="HuggingFace model ID (e.g., Qwen/Qwen2-0.5B)")
    parser.add_argument("--layer", type=str, help="Specific layer to extract (for M1)")
    args = parser.parse_args()
    
    print("Fornax Compiler Starting...")
    # TODO: Implement M1 flow
    
if __name__ == "__main__":
    main()
