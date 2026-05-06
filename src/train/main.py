import torch
from train.ablation import run_ablation_study

def main():
    print("Starting Project 2 Model and Training Execution...")
    run_ablation_study()
    print("Execution completed.")

if __name__ == "__main__":
    main()