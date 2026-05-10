# generate_data.py
from pathlib import Path
import tomli # Python 3.10+
from pytorchexample.data import generate_distributed_datasets

def main():
    # Read default hyperparameters from pyproject.toml
    config_text = Path("pyproject.toml").read_text()
    cfg = tomli.loads(config_text)["tool"]["flwr"]["app"]["config"]
    
    num_clients = int(cfg["num-clients"])
    alpha = float(cfg["alpha-dirichlet"])
    data_dir = str(cfg["data-dir"])
    
    print(f"Generating data for {num_clients} clients with alpha={alpha}...")
    generate_distributed_datasets(k=num_clients, alpha=alpha, save_dir=data_dir)

if __name__ == "__main__":
    main()