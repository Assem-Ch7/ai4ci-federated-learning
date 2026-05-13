"""pytorchexample: A Flower / PyTorch app."""

import json
import random
import time
from pathlib import Path
from typing import Dict, List

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, Message, RecordDict
from flwr.serverapp import Grid, ServerApp

# Import your custom model
from pytorchexample.model import CustomFashionModel

# Create ServerApp
app = ServerApp()

# --- Helper Functions ---

def sample_clients(node_ids, fraction: float, global_rng: random.Random) -> List[int]:
    """Sample a fraction of available clients."""
    node_ids_list = list(node_ids)
    num_to_sample = max(1, int(len(node_ids_list) * fraction))
    return global_rng.sample(node_ids_list, k=num_to_sample)

def fedavg(state_dicts: List[Dict[str, torch.Tensor]], num_examples: List[int]) -> Dict[str, torch.Tensor]:
    """Apply parameter aggregation via weighted average."""
    total_examples = sum(num_examples)
    aggregated_dict = {}
    
    for key in state_dicts[0].keys():
        weighted_sum = sum(state_dict[key] * num for state_dict, num in zip(state_dicts, num_examples))
        aggregated_dict[key] = weighted_sum / total_examples
        
    return aggregated_dict

def aggregate_metrics(metrics: List[Dict[str, float]], num_examples: List[int]) -> Dict[str, float]:
    """Apply metric aggregation via weighted average."""
    total_examples = sum(num_examples)
    aggregated_metrics = {}
    
    if not metrics:
        return aggregated_metrics
        
    for key in metrics[0].keys():
        weighted_sum = sum(metric[key] * num for metric, num in zip(metrics, num_examples))
        aggregated_metrics[key] = weighted_sum / total_examples
        
    return aggregated_metrics

# --- Manual Orchestration Loop ---

@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""
    
    # 1. Read run config
    fraction_train = float(context.run_config["fraction-train"])
    num_rounds = int(context.run_config["num-server-rounds"])
    lr = float(context.run_config["learning-rate"])
    seed = int(context.run_config["seed"])
    
    # Read algorithm toggles
    mu = float(context.run_config.get("fedprox-mu", 0.0))
    # Add a toggle in your TOML for SCAFFOLD!
    use_scaffold = bool(context.run_config.get("use-scaffold", False)) 

    global_rng = random.Random(seed)
    results = [] 
    
    # Initialize the global model
    global_model = CustomFashionModel()
    global_state_dict = global_model.state_dict()

    #  SCAFFOLD LOGIC: Initialize Global Compass 
    global_c_dict = None
    if use_scaffold:
        print("🌍 Server initializing SCAFFOLD Global Control Variate (Compass).")
        # Initialize c to all zeros, matching the shape of every model layer
        global_c_dict = {k: torch.zeros_like(v) for k, v in global_state_dict.items()}
    # ==========================================

    print(f"Starting orchestration for {num_rounds} rounds...")

    for current_round in range(1, num_rounds + 1):
        print(f"\n--- Round {current_round} ---")
        round_metrics = {"round": current_round}
        
        # A. Wait for clients and Sample
        available_node_ids = grid.get_node_ids()
        while len(available_node_ids) == 0:
            print("Server: Waiting for virtual clients to boot up...")
            time.sleep(2)
            available_node_ids = grid.get_node_ids()
            
        sampled_nodes = sample_clients(available_node_ids, fraction_train, global_rng)
        round_metrics["num_training_clients"] = len(sampled_nodes)
        
        # B. Prepare and Send "train" messages
        train_messages = []
        for node_id in sampled_nodes:
            payload = RecordDict()
            payload["arrays"] = ArrayRecord(global_state_dict) 
            payload["config"] = ConfigRecord({"lr": lr, "mu": mu})
            
            # 🏗️ Attach the Global Compass to the outgoing mail if SCAFFOLD is active
            if use_scaffold:
                payload["c_global"] = ArrayRecord(global_c_dict)
                
            msg = Message(
                message_type="train",
                dst_node_id=node_id,
                group_id=str(current_round),
                content=payload
            )
            train_messages.append(msg)
            
        # C. Collect replies from clients
        train_replies = grid.send_and_receive(train_messages)
        
        client_state_dicts = []
        client_num_examples = []
        client_metrics = []
        
        # List to hold the returning Local Compasses
        client_c_local_dicts = [] 
        
        for reply in train_replies:
            if not reply.has_content():
                print("Server: Skipping a client that crashed during training.")
                continue

            client_state_dicts.append(reply.content["arrays"].to_torch_state_dict())
            metrics = reply.content["metrics"]
            client_num_examples.append(int(metrics["num_examples"]))
            client_metrics.append({
                "train_loss": float(metrics["train_loss"]),
                "train_acc": float(metrics["train_acc"])
            })
            
            # 🏗️ Extract the returned Local Compass (c_k)
            if use_scaffold and "c_local" in reply.content:
                client_c_local_dicts.append(reply.content["c_local"].to_torch_state_dict())
                
        if not client_state_dicts:
            print("CRITICAL: All sampled clients crashed this round! Skipping aggregation.")
            continue

        # D. Aggregate updates to form the NEW global model (Standard FedAvg Math)
        global_state_dict = fedavg(client_state_dicts, client_num_examples)
        
        # SCAFFOLD LOGIC: Update Global Compass 
        if use_scaffold and client_c_local_dicts:
            num_participating = len(client_c_local_dicts)
            
            # c = c + (1/|S|) * sum(c_k - c)
            for k in global_c_dict.keys():
                # Sum the difference for the current layer
                sum_diff = sum([c_k[k] - global_c_dict[k] for c_k in client_c_local_dicts])
                # Apply the update
                global_c_dict[k] = global_c_dict[k] + (1.0 / num_participating) * sum_diff
        # ==========================================

        # E. Aggregate metrics for logging
        agg_metrics = aggregate_metrics(client_metrics, client_num_examples)
        round_metrics["train_loss"] = agg_metrics.get("train_loss", 0.0)
        round_metrics["train_acc"] = agg_metrics.get("train_acc", 0.0)
        
        print(f"Server: Evaluating global model for round {current_round}...")
        
        # 1. Sample nodes for evaluation
        fraction_eval = float(context.run_config.get("fraction-evaluate", 0.5))
        eval_nodes = sample_clients(available_node_ids, fraction_eval, global_rng)
        
        # 2. Prepare evaluation messages
        eval_messages = []
        for node_id in eval_nodes:
            payload = RecordDict()
            payload["arrays"] = ArrayRecord(global_state_dict) 
            payload["config"] = ConfigRecord({}) 
            
            eval_messages.append(Message(
                message_type="evaluate",
                dst_node_id=node_id,
                group_id=str(current_round),
                content=payload
            ))

        # 3. Send and receive evaluation results
        eval_replies = grid.send_and_receive(eval_messages)
        
        eval_metrics_list = []
        eval_num_examples = []
        for reply in eval_replies:
            if not reply.has_content():
                continue
            m = reply.content["metrics"]
            eval_num_examples.append(int(m["num_examples"]))
            eval_metrics_list.append({
                "eval_loss": float(m["eval_loss"]),
                "eval_acc": float(m["eval_acc"])
            })
            
        if not eval_metrics_list:
            continue
            
        # 4. Aggregate evaluation metrics
        agg_eval = aggregate_metrics(eval_metrics_list, eval_num_examples)
        round_metrics["eval_loss"] = agg_eval.get("eval_loss", 0.0)
        round_metrics["eval_acc"] = agg_eval.get("eval_acc", 0.0)

        print(f"Round {current_round} Eval Loss: {round_metrics['eval_loss']:.4f}")
        print(f"Round {current_round} Eval Acc: {round_metrics['eval_acc']:.4f}")

        # Append metrics to results
        results.append(round_metrics)
        
    # Save results to JSON
    save_path = Path("results")
    save_path.mkdir(parents=True, exist_ok=True)
    run_id = context.run_id
    
    with open(save_path / f"{run_id}.json", "w") as f:
        json.dump(results, f, indent=4)
        
    print(f"\nSimulation complete. Results saved to results/{run_id}.json")