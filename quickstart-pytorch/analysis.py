# analysis.py
from __future__ import annotations
from typing import Dict, List
import json
from pathlib import Path
import matplotlib.pyplot as plt
from prettytable import PrettyTable

class ResultsVisualizer:
    """
    Multi-run analyzer:
    - Each run has a name/label (e.g., "alpha=0.1", "alpha=1.0")
    - Each run points to one JSON file produced by the server
    """
    def __init__(self) -> None:
        # runs[name] = list of per-round dicts loaded from JSON
        self.runs: Dict[str, List[dict]] = {}

    def add_run(self, name: str, file_name: str) -> None:
        """
        Load a JSON results file and register it under 'name'.
        """
        file_path = Path(file_name)
        if not file_path.exists():
            print(f"Error: File {file_name} not found.")
            return
            
        with open(file_path, "r") as f:
            data = json.load(f)
            self.runs[name] = data
            print(f"Successfully loaded run: '{name}' ({len(data)} rounds)")

    def print_run_summary_table(self) -> None:
        """
        Print a table where each row is one run, containing:
        - run name
        - final train_loss, final train_acc
        - final eval_loss, final eval_acc
        """
        table = PrettyTable()
        table.field_names = ["Run Name", "Final Train Loss", "Final Train Acc", "Final Eval Loss", "Final Eval Acc"]
        
        for name, rounds_data in self.runs.items():
            if not rounds_data:
                continue
            
            final_round = rounds_data[-1]
            
            # Check for multiple possible naming conventions
            t_loss = final_round.get("train_loss", 0.0)
            t_acc = final_round.get("train_acc", 0.0)
            
            # This looks for 'eval_loss', 'fed_eval_loss', or 'eval_loss_aggregated'
            e_loss = final_round.get("eval_loss", final_round.get("fed_eval_loss", 0.0))
            e_acc = final_round.get("eval_acc", final_round.get("fed_eval_acc", 0.0))
            
            table.add_row([name, round(t_loss, 4), round(t_acc, 4), round(e_loss, 4), round(e_acc, 4)])
            
        print("\n=== Simulation Summary ===")
        print(table)

    def plot_metric(self, metric: str, fig_directory: str) -> None:
        """
        Plot 'metric' vs 'round' for all runs on the same figure.
        """
        save_dir = Path(fig_directory)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        plt.figure(figsize=(8, 5))
        data_plotted = False
        
        for name, rounds_data in self.runs.items():
            # Only plot rounds that actually contain the requested metric
            rounds = [r.get("round") for r in rounds_data if metric in r]
            values = [r.get(metric) for r in rounds_data if metric in r]
            
            if rounds and values:
                plt.plot(rounds, values, marker='o', label=name)
                data_plotted = True
                
        if not data_plotted:
            print(f"Warning: Metric '{metric}' not found in any runs. Skipping.")
            plt.close()
            return
            
        plt.title(f"Comparative Learning Curve: {metric}")
        plt.xlabel("Communication Round")
        plt.ylabel(metric)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        save_path = save_dir / f"{metric}.png"
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        print(f"Saved plot: {save_path}")

    def plot_all(self, fig_directory: str) -> None:
        """
        Convenience method that plots a set of standard metrics.
        """
        # The metrics your server app is currently saving to the JSON
        standard_metrics = ["train_loss", "train_acc", "eval_loss", "eval_acc"]
        
        print(f"\nGenerating comparative plots in '{fig_directory}'...")
        for metric in standard_metrics:
            self.plot_metric(metric, fig_directory)




if __name__ == "__main__":
    # 1. Initialize the analyzer
    analyzer = ResultsVisualizer()
    
    # 2. Add all your experimental runs
    # --- Set 1: Heterogeneity (Alpha) ---
    #analyzer.add_run("Alpha 0.1 (Non-IID)", "results/vary_alpha/alpha_0.1.json")
    #analyzer.add_run("Alpha 5.0", "results/vary_alpha/alpha_5.json")
    #analyzer.add_run("Alpha 100.0 (IID)", "results/vary_alpha/alpha_100.json")
    
    # --- Set 2: Scale (Number of Clients) ---
    #analyzer.add_run("Clients: 5", "results/vary_clients/clients_5.json")
    #analyzer.add_run("Clients: 20", "results/vary_clients/clients_20.json")
    # Note: Your Alpha 0.1 run also counts as the "50 Clients" run
    
    # --- Set 3: Efficiency (Sampling Fraction) ---
    #analyzer.add_run("Sampling: 10%", "results/vary_fraction/fraction_0.1.json")
    #analyzer.add_run("Sampling: 50%", "results/vary_fraction/fraction_0.5.json")
    # Note: Your Alpha 0.1 run also counts as "100% Sampling"

    # Load the Step 8 runs
    analyzer.add_run("FedAvg (SGD)", "results/vary_algo/fedavg_sgd.json")
    analyzer.add_run("FedAvg (Adam)", "results/vary_algo/fedavg_adam.json")
    analyzer.add_run("FedSGD (30 Rounds)", "results/vary_algo/fedsgd.json") #Calculated equivalence required 113 rounds for FedSGD

    # 3. Print the master summary table
    analyzer.print_run_summary_table()
    
    # 4. Generate the comparative plots
    # This will overlay ALL runs on one graph, which might be messy.
    # If you want separate plots for each experiment, you'd call plot_metric 
    # multiple times with filtered data. 
    analyzer.plot_all("results/figures")
    
    print("\nCheck the 'results/figures' folder for your comparison curves!")