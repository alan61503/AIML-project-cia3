# Step 0: Install PyTorch Geometric if not installed
# !pip install torch torch_geometric torch_scatter torch_sparse torch_cluster torch_spline_conv

import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
try:
    # PyG >= 2.3
    from torch_geometric.explain.algorithm import GNNExplainer  # type: ignore
    _EXPLAIN_AVAILABLE = True
except Exception:
    try:
        # Older PyG
        from torch_geometric.nn.models import GNNExplainer  # type: ignore
        _EXPLAIN_AVAILABLE = True
    except Exception:
        _EXPLAIN_AVAILABLE = False
from torch_geometric.utils import dropout_adj
import random
import numpy as np
import argparse
from typing import Dict, Any, Optional, Tuple
import os
import matplotlib
matplotlib.use('Agg')  # non-interactive backend to avoid blocking
import matplotlib.pyplot as plt

try:
    from sklearn.metrics import f1_score
    _SKLEARN_AVAILABLE = True
except Exception:
    _SKLEARN_AVAILABLE = False

# -----------------------
# Step 1: Set seeds for reproducibility
# -----------------------
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# -----------------------
# Step 2: Load Dataset
# -----------------------
def username_safe(name: str) -> str:
    return name.strip()

def load_dataset(dataset_name: str):
    dataset = Planetoid(root=f'./data/{dataset_name}', name=username_safe(dataset_name))
    return dataset, dataset[0]

# -----------------------
# Step 3: Optional - Add Noise
# -----------------------
def add_feature_noise(data, noise_level=0.1):
    noisy_data = data.clone()
    if hasattr(noisy_data, 'x') and noisy_data.x is not None:
        noise = torch.randn_like(noisy_data.x) * noise_level
        noisy_data.x = noisy_data.x + noise
    return noisy_data

def feature_dropout(data, drop_prob: float = 0.0):
    if drop_prob <= 0:
        return data
    d = data.clone()
    if hasattr(d, 'x') and d.x is not None:
        mask = torch.rand_like(d.x) > drop_prob
        d.x = d.x * mask
    return d

def edge_dropout(data, drop_prob: float = 0.0):
    if drop_prob <= 0:
        return data
    d = data.clone()
    edge_index, _ = dropout_adj(d.edge_index, p=drop_prob, force_undirected=False,
                                num_nodes=d.num_nodes, training=True)
    d.edge_index = edge_index
    return d

def apply_augmentations(data, noise_level: float, feat_drop: float, edge_drop: float):
    d = add_feature_noise(data, noise_level=noise_level) if noise_level > 0 else data
    d = feature_dropout(d, drop_prob=feat_drop)
    d = edge_dropout(d, drop_prob=edge_drop)
    return d

# -----------------------
# Step 4: Define Models
# -----------------------

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x

class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=8, dropout=0.6):
        super(GAT, self).__init__()
        self.gat1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout)
        self.gat2 = GATConv(hidden_channels * heads, out_channels, heads=1,
                            concat=False, dropout=dropout)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gat1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gat2(x, edge_index)
        return x

class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.lin = torch.nn.Linear(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin(x)
        return x

# -----------------------
# Step 5: Initialize Model
# -----------------------
def build_model(model_choice: str, in_channels: int, hidden_channels: int, out_channels: int, dropout: float):
    if model_choice == 'gcn':
        return GCN(in_channels, hidden_channels, out_channels, dropout=dropout)
    elif model_choice == 'gat':
        return GAT(in_channels, hidden_channels, out_channels, dropout=dropout)
    elif model_choice == 'sage':
        return GraphSAGE(in_channels, hidden_channels, out_channels, dropout=dropout)
    else:
        raise ValueError(f"Unknown model_choice: {model_choice}")

criterion = torch.nn.CrossEntropyLoss()

# -----------------------
# Step 6: Train Function
# -----------------------
def train(model, data, optimizer, epochs=200, log_every=20):
    model.train()
    last_loss = None
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        last_loss = float(loss.detach().cpu())
        if log_every and epoch % log_every == 0:
            acc, _ = evaluate(model, data)
            print(f'Epoch {epoch:03d}, Loss: {last_loss:.4f}, Test Accuracy: {acc:.4f}')
    return last_loss

# -----------------------
# Step 7: Test Function
# -----------------------
@torch.no_grad()
def evaluate(model, data) -> Tuple[float, Optional[float]]:
    model.eval()
    logits = model(data.x, data.edge_index)
    pred = logits.argmax(dim=1)
    correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
    acc = int(correct) / int(data.test_mask.sum())
    f1 = None
    if _SKLEARN_AVAILABLE:
        y_true = data.y[data.test_mask].detach().cpu().numpy()
        y_pred = pred[data.test_mask].detach().cpu().numpy()
        try:
            f1 = float(f1_score(y_true, y_pred, average='macro'))
        except Exception:
            f1 = None
    return acc, f1

# -----------------------
# Step 7b: Explainability
# -----------------------
def explain_model(model, data, node_idx=None, model_name: str = 'model'):
    if not _EXPLAIN_AVAILABLE:
        print("Explainability not available: GNNExplainer not found in this PyG version.")
        # Fall back to gradient-based saliency
    model.eval()
    if node_idx is None:
        node_idx = random.choice(torch.where(data.test_mask)[0].tolist())
    print(f"\nExplaining prediction for node {node_idx}...")
    os.makedirs('explanations', exist_ok=True)

    used_gnn_explainer = False
    if _EXPLAIN_AVAILABLE:
        try:
            # Instantiate explainer with version fallback
            try:
                explainer = GNNExplainer(model, epochs=200)
            except TypeError:
                try:
                    explainer = GNNExplainer(model)
                except TypeError:
                    explainer = GNNExplainer(epochs=200)
            # Attempt common method names
            if hasattr(explainer, 'explain_node'):
                try:
                    node_feat_mask, edge_mask = explainer.explain_node(node_idx, data.x, data.edge_index)
                except TypeError:
                    node_feat_mask, edge_mask = explainer.explain_node(node_idx, model, data.x, data.edge_index)
                used_gnn_explainer = True
                try:
                    explainer.visualize_subgraph(node_idx, data.edge_index, edge_mask, y=data.y)
                    out_path = os.path.join('explanations', f'{model_name}_node{node_idx}_subgraph.png')
                    plt.savefig(out_path, dpi=150, bbox_inches='tight')
                    plt.close()
                    print(f"Saved subgraph explanation to {out_path}")
                except Exception as e:
                    print(f"Visualization not available: {e}")
        except Exception as e:
            print(f"GNNExplainer failed, falling back to saliency: {e}")

    if not used_gnn_explainer:
        # Gradient-based feature saliency as fallback
        x = data.x.clone().detach().requires_grad_(True)
        logits = model(x, data.edge_index)
        pred_class = int(logits[node_idx].argmax())
        target = logits[node_idx, pred_class]
        model.zero_grad(set_to_none=True)
        target.backward()
        saliency = x.grad[node_idx].abs().detach().cpu()
        topk = min(10, saliency.shape[0])
        top_idx = torch.topk(saliency, k=topk).indices.tolist()
        print(f"Predicted class: {pred_class}")
        print(f"Top-{topk} salient feature indices: {top_idx}")
        try:
            plt.figure(figsize=(6,3))
            vals = saliency[top_idx].numpy()
            plt.bar(range(len(top_idx)), vals)
            plt.xticks(range(len(top_idx)), top_idx, rotation=45)
            plt.title('Feature saliency (|d logit / d x|)')
            plt.tight_layout()
            out_path = os.path.join('explanations', f'{model_name}_node{node_idx}_saliency.png')
            plt.savefig(out_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Saved saliency plot to {out_path}")
        except Exception as e:
            print(f"Could not save saliency plot: {e}")

# -----------------------
# Step 8: Experiment Runner
# -----------------------
def run_experiment(args: argparse.Namespace) -> Dict[str, Any]:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    dataset, data = load_dataset(args.dataset)
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    data = data.to(device)

    in_channels = dataset.num_node_features
    out_channels = dataset.num_classes
    model = build_model(args.model, in_channels, args.hidden_channels, out_channels,
                        dropout=args.dropout).to(device)

    aug_data = apply_augmentations(data, noise_level=args.noise,
                                   feat_drop=args.feature_dropout, edge_drop=args.edge_dropout)
    aug_data = aug_data.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    final_loss = train(model, aug_data, optimizer, epochs=args.epochs, log_every=args.log_every)
    acc, f1 = evaluate(model, data)

    # Explainability (optional)
    if args.explain:
        explain_model(model, data)

    result: Dict[str, Any] = {
        'acc': acc,
        'f1': f1,
        'final_loss': final_loss,
        'device': str(device)
    }
    if getattr(args, 'return_model', False):
        result['model'] = model
        result['data'] = data
    return result

# -----------------------
# Step 8b: Run all models in one go
# -----------------------
def run_all_models(args: argparse.Namespace):
    models = ['gcn', 'gat', 'sage']
    results = []
    print("\n=== Running all models (GCN, GAT, SAGE) ===")
    for m in models:
        cfg = argparse.Namespace(**{**vars(args), 'model': m, 'explain': False, 'return_model': True})
        print(f"\nTraining {m.upper()}...")
        metrics = run_experiment(cfg)
        results.append((m, metrics))
        if getattr(args, 'explain', False):
            try:
                explain_model(metrics['model'], metrics['data'], model_name=m)
            except Exception as e:
                print(f"Explainability for {m.upper()} skipped: {e}")
    print("\nModel, Clean Accuracy, Clean Macro-F1")
    for m, met in results:
        acc = f"{met['acc']:.4f}" if met.get('acc') is not None else "-"
        f1 = f"{met['f1']:.4f}" if met.get('f1') is not None else "-"
        print(f"{m},{acc},{f1}")

# -----------------------
# Step 9: CLI
# -----------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='GNN on Planetoid datasets with explainability')
    parser.add_argument('--dataset', type=str, default='Cora', choices=['Cora', 'CiteSeer', 'PubMed'])
    parser.add_argument('--model', type=str, default='gcn', choices=['gcn', 'gat', 'sage'])
    parser.add_argument('--hidden-channels', dest='hidden_channels', type=int, default=16)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--noise', type=float, default=0.1)
    parser.add_argument('--feature-dropout', dest='feature_dropout', type=float, default=0.0)
    parser.add_argument('--edge-dropout', dest='edge_dropout', type=float, default=0.0)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--log-every', dest='log_every', type=int, default=20)
    parser.add_argument('--explain', action='store_true', help='Run GNNExplainer on a random test node')
    parser.add_argument('--all-models', action='store_true', help='Run GCN, GAT, and SAGE sequentially')
    return parser.parse_args()

def main():
    args = parse_args()
    if args.all_models:
        run_all_models(args)
        return
    metrics = run_experiment(args)
    print(f"\nDevice: {metrics['device']}")
    print(f"Clean Accuracy: {metrics['acc']:.4f}")
    if metrics['f1'] is not None:
        print(f"Clean Macro-F1: {metrics['f1']:.4f}")

if __name__ == '__main__':
    main()
