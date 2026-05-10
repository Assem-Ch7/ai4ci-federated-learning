"""pytorchexample: A Flower / PyTorch app."""

import torch
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
    
    # Read the learning rate sent by the server in the message payload 
    lr = float(msg.content["config"].get("lr", 0.01))

    # 2. Receive the global model weights from msg 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CustomFashionModel().to(device)
    
    # Load weights (unpack ArrayRecord to PyTorch state_dict)
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    
    # 3. Load the client dataset using partition_id [cite: 335]
    train_loader, _ = load_client_data(cid=partition_id, data_dir=data_dir, batch_size=batch_size)
    
    # 4. Train locally for local-epochs 
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()
    
    avg_loss, accuracy = 0.0, 0.0
    for _ in range(local_epochs):
        avg_loss, accuracy = model.train_epoch(train_loader, criterion, optimizer, device)
        
    # 5. Return the updated weights and training metrics in a message 
    out_payload = RecordDict()
    out_payload["arrays"] = ArrayRecord(model.state_dict())
    out_payload["metrics"] = ConfigRecord({
        "train_loss": avg_loss,
        "train_acc": accuracy,
        "num_examples": len(train_loader.dataset) # Required metric 
    })
    
    return Message(content=out_payload, reply_to=msg)

# --- Task 4.2: Client-side evaluation ---
@app.evaluate()
def evaluate(msg: Message, context: Context) -> Message:
    """Evaluate the model locally."""
    print("Client: Received evaluate message")
    
    # 1. Get partition ID and config 
    partition_id = int(context.node_config["partition-id"])
    data_dir = context.run_config.get("data-dir", "./data")
    batch_size = int(context.run_config.get("batch-size", 32))
    
    # 2. Receive global weights
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CustomFashionModel().to(device)
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    
    # 3. Load the validation dataset 
    _, val_loader = load_client_data(cid=partition_id, data_dir=data_dir, batch_size=batch_size)
    
    # 4. Evaluate the model locally 
    criterion = torch.nn.CrossEntropyLoss()
    avg_loss, accuracy = model.test_epoch(val_loader, criterion, device)
    
    # 5. Return evaluation metrics 
    out_payload = RecordDict()
    out_payload["metrics"] = ConfigRecord({
        "eval_loss": avg_loss,
        "eval_acc": accuracy,
        "num_examples": len(val_loader.dataset) # Required metric 
    })
    
    return Message(content=out_payload, reply_to=msg)