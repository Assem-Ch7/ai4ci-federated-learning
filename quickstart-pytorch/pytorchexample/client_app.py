"""pytorchexample: A Flower / PyTorch app."""

import torch
import copy
from flwr.client import ClientApp
from flwr.common import Context, Message, RecordDict, ArrayRecord, ConfigRecord

from pytorchexample.model import CustomFashionModel
from pytorchexample.data import load_client_data

# Create ClientApp
app = ClientApp()

# --- Task 4.1: Client-side training ---
@app.train()
def train(msg: Message, context: Context) -> Message:
    """Train the model locally."""
    print("Client: Received train message")
    
    # 1. Get partition ID and hyperparameters from context 
    partition_id = int(context.node_config["partition-id"])
    data_dir = context.run_config.get("data-dir", "./data")
    local_epochs = int(context.run_config.get("local-epochs", 1))
    batch_size = int(context.run_config.get("batch-size", 32))
    
    # Read the learning rate and proximal parameter 
    lr = float(msg.content["config"].get("lr", 0.01))
    mu = float(msg.content["config"].get("mu", 0.0)) 

    # 2. Load the client dataset
    train_loader, _ = load_client_data(cid=partition_id, data_dir=data_dir, batch_size=batch_size)
    
    if train_loader is None:
        print(f"Client {partition_id}: No data received! Skipping training.")
        out_payload = RecordDict()
        out_payload["arrays"] = msg.content["arrays"] 
        out_payload["metrics"] = ConfigRecord({"train_loss": 0.0, "train_acc": 0.0, "num_examples": 0})
        return Message(content=out_payload, reply_to=msg) 

    # 3. NORMAL TRAINING SETUP
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CustomFashionModel().to(device)
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    
    # Create an exact, independent copy of the starting weights
    global_model = copy.deepcopy(model)
    for param in global_model.parameters():
        param.requires_grad = False
    global_model.eval()

    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()
    
    #  SCAFFOLD PRE-TRAINING LOGIC (Memory Load) 
    # Check if the server sent a global compass
    is_scaffold = "c_global" in msg.content
    
    # We explicitly extract named parameters to ensure our compasses align with weights perfectly
    param_keys = list(dict(model.named_parameters()).keys())
    
    if is_scaffold:
        # Load c_global from the server
        c_global_dict = msg.content["c_global"].to_torch_state_dict()
        c_global = [c_global_dict[k].to(device) for k in param_keys]
        
        # Load c_local from persistent state, or initialize to zeros
        if "c_local" in context.state:
            c_local_dict = context.state["c_local"].to_torch_state_dict()
            c_local = [c_local_dict[k].to(device) for k in param_keys]
        else:
            c_local = [torch.zeros_like(p, device=device) for p in model.parameters()]
    else:
        c_global = None
        c_local = None
        
    T = local_epochs * len(train_loader) # Total steps needed for the formula later
    # ==========================================

    # 4. TRAINING LOOP
    avg_loss, accuracy = 0.0, 0.0
    for _ in range(local_epochs):
        # We pass the new SCAFFOLD compass variables into your updated train_epoch function
        avg_loss, accuracy = model.train_epoch(
            train_loader, criterion, optimizer, device,
            global_model=global_model, mu=mu,
            c_global=c_global, c_local=c_local
        )
        
    # SCAFFOLD POST-TRAINING LOGIC (Memory Save)

    c_local_new_dict = {}
    if is_scaffold:
        for k, c_g, c_k, w_t, w_t1 in zip(param_keys, c_global, c_local, global_model.parameters(), model.parameters()):
            # The SCAFFOLD Formula: c_k_new = c_k - c + (1 / (lr * T)) * (w_t - w_t+1)
            new_c_k = c_k - c_g + (1.0 / (lr * T)) * (w_t - w_t1)
            c_local_new_dict[k] = new_c_k.cpu() # Move to CPU for safe storage
            
        # Save the new local compass back to the client's hard drive so it remembers it next round!
        context.state["c_local"] = ArrayRecord(c_local_new_dict)
    # ==========================================
        
    # 5. Pack results and return
    out_payload = RecordDict()
    out_payload["arrays"] = ArrayRecord(model.state_dict())
    
    # Send the updated local compass back to the server so it can update the global compass
    if is_scaffold:
        out_payload["c_local"] = ArrayRecord(c_local_new_dict)
        
    out_payload["metrics"] = ConfigRecord({
        "train_loss": avg_loss,
        "train_acc": accuracy,
        "num_examples": len(train_loader.dataset)
    })
    
    return Message(content=out_payload, reply_to=msg)

# --- Task 4.2: Client-side evaluation ---
@app.evaluate()
def evaluate(msg: Message, context: Context) -> Message:
    """Evaluate the model locally."""
    print("Client: Received evaluate message")
    
    partition_id = int(context.node_config["partition-id"])
    data_dir = context.run_config.get("data-dir", "./data")
    batch_size = int(context.run_config.get("batch-size", 32))
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CustomFashionModel().to(device)
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    
    _, val_loader = load_client_data(cid=partition_id, data_dir=data_dir, batch_size=batch_size)
    
    if val_loader is None:
        print(f"Client {partition_id}: No eval data! Skipping evaluation.")
        out_payload = RecordDict()
        out_payload["metrics"] = ConfigRecord({"eval_loss": 0.0, "eval_acc": 0.0, "num_examples": 0})
        return Message(content=out_payload, reply_to=msg)
        
    criterion = torch.nn.CrossEntropyLoss()
    avg_loss, accuracy = model.test_epoch(val_loader, criterion, device)
    
    out_payload = RecordDict()
    out_payload["metrics"] = ConfigRecord({
        "eval_loss": avg_loss,
        "eval_acc": accuracy,
        "num_examples": len(val_loader.dataset) 
    })
    
    return Message(content=out_payload, reply_to=msg)