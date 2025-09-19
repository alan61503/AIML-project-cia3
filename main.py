# Step 0: Install PyTorch Geometric if not installed
# !pip install torch torch_geometric torch_scatter torch_sparse torch_cluster torch_spline_conv

import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
from torch_geometric.utils import dropout_adj
import random
import numpy as np
import argparse
from typing import Dict, Any, Optional, Tuple

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
    edge_index, _ = dropout_adj(d.edge_index, p=drop_prob, force_undirected=False, num_nodes=d.num_nodes, training=True)
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

# ----- GCN Model -----
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

# ----- GAT Model -----
class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=8, dropout=0.6):
        super(GAT, self).__init__()
        self.gat1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout)
        self.gat2 = GATConv(hidden_channels * heads, out_channels, heads=1, concat=False, dropout=dropout)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gat1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gat2(x, edge_index)
        return x

# ----- GraphSAGE Model -----
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

def username_safe(name: str) -> str:
    # Planetoid expects canonical names; keep common variants safe
    return name.strip()

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

def run_experiment(args: argparse.Namespace) -> Dict[str, Any]:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    dataset, data = load_dataset(args.dataset)
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    data = data.to(device)

    in_channels = dataset.num_node_features
    out_channels = dataset.num_classes
    model = build_model(args.model, in_channels, args.hidden_channels, out_channels, dropout=args.dropout).to(device)

    aug_data = apply_augmentations(data, noise_level=args.noise, feat_drop=args.feature_dropout, edge_drop=args.edge_dropout)
    aug_data = aug_data.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    final_loss = train(model, aug_data, optimizer, epochs=args.epochs, log_every=args.log_every)
    acc, f1 = evaluate(model, data)
    acc_noisy = None
    f1_noisy = None
    if getattr(args, 'eval_noisy', False):
        noisy_eval = add_feature_noise(data, noise_level=args.eval_noise if args.eval_noise is not None else args.noise)
        noisy_eval = noisy_eval.to(device)
        acc_noisy, f1_noisy = evaluate(model, noisy_eval)
    return {
        'acc': acc,
        'f1': f1,
        'acc_noisy': acc_noisy,
        'f1_noisy': f1_noisy,
        'final_loss': final_loss,
        'device': str(device)
    }

def grid_search(args: argparse.Namespace):
    best: Dict[str, Any] = {'score': -1.0, 'params': None}
    hidden_list = args.gs_hidden or [args.hidden_channels]
    lr_list = args.gs_lr or [args.lr]
    drop_list = args.gs_dropout or [args.dropout]
    for h in hidden_list:
        for lr in lr_list:
            for dr in drop_list:
                cfg = argparse.Namespace(**{**vars(args), 'hidden_channels': h, 'lr': lr, 'dropout': dr})
                print(f"\nGrid run: hidden={h}, lr={lr}, dropout={dr}")
                metrics = run_experiment(cfg)
                score = metrics['acc']
                if score > best['score']:
                    best = {'score': score, 'params': {'hidden_channels': h, 'lr': lr, 'dropout': dr}, 'metrics': metrics}
    print("\nBest config:", best['params'])
    print(f"Accuracy: {best['metrics']['acc']:.4f}", end='')
    if best['metrics']['f1'] is not None:
        print(f", Macro-F1: {best['metrics']['f1']:.4f}")
    else:
        print()

def compare_models(args: argparse.Namespace):
    models = args.compare_models or []
    if not models:
        print("No models provided for comparison.")
        return
    results = []
    print("\n=== Multi-model comparison ===")
    for m in models:
        cfg = argparse.Namespace(**{**vars(args), 'model': m})
        print(f"\nTraining {m.upper()}...")
        metrics = run_experiment(cfg)
        results.append((m, metrics))
    # Print summary
    print("\nModel, Clean Accuracy, Clean Macro-F1, Noisy Accuracy, Noisy Macro-F1")
    for m, met in results:
        acc = f"{met['acc']:.4f}" if met.get('acc') is not None else "-"
        f1 = f"{met['f1']:.4f}" if met.get('f1') is not None else "-"
        accn = f"{met['acc_noisy']:.4f}" if met.get('acc_noisy') is not None else "-"
        f1n = f"{met['f1_noisy']:.4f}" if met.get('f1_noisy') is not None else "-"
        print(f"{m},{acc},{f1},{accn},{f1n}")

# -----------------------
# Step 8: Run Training on Clean Data
# -----------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='GNN on Planetoid datasets with simple augmentations and tuning')
    parser.add_argument('--dataset', type=str, default='Cora', choices=['Cora', 'CiteSeer', 'PubMed'], help='Dataset name')
    parser.add_argument('--model', type=str, default='gcn', choices=['gcn', 'gat', 'sage'], help='Model variant')
    parser.add_argument('--hidden-channels', dest='hidden_channels', type=int, default=16, help='Hidden channels')
    parser.add_argument('--dropout', type=float, default=0.5, help='Model dropout')
    parser.add_argument('--epochs', type=int, default=200, help='Training epochs')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=5e-4, help='Weight decay')
    parser.add_argument('--noise', type=float, default=0.1, help='Gaussian feature noise std for training augmentation')
    parser.add_argument('--feature-dropout', dest='feature_dropout', type=float, default=0.0, help='Feature dropout prob')
    parser.add_argument('--edge-dropout', dest='edge_dropout', type=float, default=0.0, help='Edge dropout prob')
    parser.add_argument('--eval-noisy', dest='eval_noisy', action='store_true', help='Also evaluate on noisy features')
    parser.add_argument('--eval-noise', dest='eval_noise', type=float, default=None, help='Noise std used only for noisy evaluation (defaults to --noise)')
    parser.add_argument('--cpu', action='store_true', help='Force CPU even if CUDA is available')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--log-every', dest='log_every', type=int, default=20, help='Log every N epochs (0 to disable)')
    parser.add_argument('--grid-search', dest='grid_search', action='store_true', help='Run simple grid search')
    parser.add_argument('--gs-hidden', dest='gs_hidden', type=int, nargs='*', default=None, help='Grid values for hidden channels')
    parser.add_argument('--gs-lr', dest='gs_lr', type=float, nargs='*', default=None, help='Grid values for learning rate')
    parser.add_argument('--gs-dropout', dest='gs_dropout', type=float, nargs='*', default=None, help='Grid values for dropout')
    parser.add_argument('--compare', dest='compare', action='store_true', help='Compare multiple models in one run')
    parser.add_argument('--compare-models', dest='compare_models', type=str, nargs='*', default=None, help='Models to compare, e.g., gcn gat sage')
    return parser.parse_args()

def main():
    args = parse_args()
    if args.grid_search:
        grid_search(args)
    elif args.compare:
        compare_models(args)
    else:
        metrics = run_experiment(args)
        print(f"\nDevice: {metrics['device']}")
        print(f"Clean Accuracy: {metrics['acc']:.4f}")
        if metrics['f1'] is not None:
            print(f"Clean Macro-F1: {metrics['f1']:.4f}")
        if metrics.get('acc_noisy') is not None:
            print(f"Noisy Accuracy: {metrics['acc_noisy']:.4f}")
            if metrics.get('f1_noisy') is not None:
                print(f"Noisy Macro-F1: {metrics['f1_noisy']:.4f}")

if __name__ == '__main__':
    main()
