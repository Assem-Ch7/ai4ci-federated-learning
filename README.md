# Federated Learning: Mitigating Client Drift with Advanced Optimization

This repository contains a PyTorch and Flower-based simulation environment for testing Federated Learning algorithms across highly heterogeneous (Non-IID) datasets. 

Developed as part of the Master's in Artificial Intelligence for Connected Industries (AI4CI) curriculum at CNAM Paris.

## 🎯 Project Objectives
The core focus of this project is to analyze and mitigate **Client Drift** caused by data heterogeneity. 

* **Phase 1 (TP1):** Implementing a manual Federated Averaging (FedAvg) pipeline with custom server-side orchestration and evaluation loops. Analyzes the impact of the Dirichlet distribution ($\alpha$) on data partitioning.
* **Phase 2 (TP2):** Extending the baseline to implement **FedProx** (proximal regularization) and **SCAFFOLD** (control variates) to stabilize training in extreme Non-IID environments.

## ⚙️ Architecture
* **Frameworks:** PyTorch, Flower (`flwr[simulation]`)
* **Dataset:** FashionMNIST (Distributed via Dirichlet partitioning)
* **Topology:** 1 Server orchestrating up to 50 simulated edge clients via Ray backend.

## How to Run

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   '''
2. **Generate the partitioned dataset:
   Modify the alpha-dirichlet parameter in pyproject.toml, then run:
   '''bash
   python3 generate_data.py
   '''
3. **Start the simulation:
   '''bash
   flwr run ./
   flwr log id (to see the logs)
   '''
4. **Analyze the results:
   '''bash
   python3 analysis.py
   '''
