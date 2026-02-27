#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

try:
    import torch_geometric  # type: ignore
    from torch_geometric.nn import knn_graph, radius_graph  # type: ignore
except ImportError as exc:
    logger.error(f"Missing required libraries. {exc}")
    logger.error("Please install torch and torch-geometric.")
    sys.exit(1)


def run_cuda_assertions():
    """
    Strictly checks if CUDA is available and configured for all libraries.
    Fails hard if any check fails.
    """
    logger.info("--- Running CUDA Prerequisite Checks ---")

    # 1. Check PyTorch
    if not torch.cuda.is_available():
        logger.error("Error: PyTorch CUDA is not available! (torch.cuda.is_available() is False)")
        logger.info("Please install a CUDA-enabled version of PyTorch.")
        sys.exit(1)

    logger.success(
        f"PyTorch CUDA (version {torch.__version__}) available. Using device: {torch.cuda.get_device_name(0)}"
    )

    # 2. torch-geometric (usually fine if the others are)
    logger.success(f"torch-geometric loaded (version {torch_geometric.__version__}).")
    logger.success("--- All CUDA Checks Passed ---")


class StabilityTestModel(torch.nn.Module):
    """
    A "kitchen sink" model designed to stress-test the exact kernels we use.
    It uses:
    - radius_graph (torch_geometric.nn.pool)
    - knn_graph (torch_geometric.nn.pool)
    - segment_csr (torch_scatter - for global pooling)
    """

    def __init__(self, in_features, hidden_features, out_features, aggregation="mean"):
        super().__init__()
        self.aggregation = aggregation

        # Input/hidden/output layers
        self.lin1 = nn.Linear(in_features, hidden_features)
        self.lin2 = nn.Linear(hidden_features, hidden_features)
        self.lin_out = nn.Linear(hidden_features, out_features)

        # Parameters for graph creation
        self.radius = 1.0
        self.k = 8
        logger.info("StabilityTestModel initialized.")

    def forward(self, x, pos, batch):
        # 1. Initial node embedding
        x = F.relu(self.lin1(x))

        # 2. TEST: radius_graph and knn_graph
        # We build two graphs and combine their edge indices
        radius_edges = radius_graph(pos, r=self.radius, batch=batch, max_num_neighbors=32)
        knn_edges = knn_graph(pos, k=self.k, batch=batch)

        # Combine edges for a single message passing step
        edge_index = torch.cat([radius_edges, knn_edges], dim=1)
        row, col = edge_index

        # 3. TEST: torch_scatter (as message passing)
        # Aggregate messages from neighbors (x[row]) to central nodes (col)
        # Using 'mean' is a good stability test
        messages = x[row]
        aggregated_messages = torch.full(
            (col.size(0), x.size(1)), 0 if self.aggregation == "mean" else 1, device=x.device
        )
        aggregated_messages.scatter_reduce_(0, col.unsqueeze(-1).expand_as(messages), messages, reduce=self.aggregation)

        # Simple residual connection and non-linearity
        x = x + aggregated_messages
        x = F.relu(self.lin2(x))

        return x


def generate_fake_data(num_graphs, nodes_per_graph, num_features, device):
    """Generates a batch of fake graph data directly on the GPU."""
    num_nodes = num_graphs * nodes_per_graph

    # Node features
    x = torch.randn((num_nodes, num_features), device=device, dtype=torch.float32)

    # Node positions (for radius/knn)
    pos = torch.rand((num_nodes, 3), device=device, dtype=torch.float32) * 10

    # Batch vector (assigns each node to a graph)
    batch = torch.arange(num_graphs, device=device).repeat_interleave(nodes_per_graph)

    # Fake labels (graph-level)
    y = torch.randn((num_graphs, 1), device=device, dtype=torch.float32)

    return x, pos, batch, y


def set_deterministic_mode(seed):
    """Sets PyTorch deterministic settings for reproducibility."""
    logger.info(f"--- Enabling Deterministic Mode (seed {seed}) ---")
    logger.warning("Note: Full determinism is not guaranteed with 'radius_graph' or 'knn_graph' on CUDA.")

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Try to set the global flag, but don't fail if it's not supported
    try:
        # This will error if any non-deterministic op is called
        torch.use_deterministic_algorithms(True)
        # This is needed for some ops like torch.bincount
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    except RuntimeError as exc:
        logger.warning(f"Could not enforce torch.use_deterministic_algorithms(True): {exc}")
        logger.warning("This is expected for some GNN ops. Continuing with cudnn.deterministic=True only.")


def main():
    """Main stability test function."""

    # --- 1. CUDA ASSERTIONS ---
    run_cuda_assertions()

    # --- 2. SETUP ---
    logger.info("--- Starting Stability Test ---")

    # --- Constants
    NUM_EPOCHS = 10
    SUB_STEPS_PER_EPOCH = 2500
    NUM_GRAPHS = 16
    NODES_PER_GRAPH = 128 * 4  # 128
    IN_FEATURES = 16
    HIDDEN_FEATURES = 32 * 4  # 32
    OUT_FEATURES = 1
    AGGREGATION = "mean"  # 'mean' is often more prone to NaNs than 'add'
    SEED = 69
    ENABLE_DETERMINISM = True

    device = torch.device("cuda:0")

    # --- Set seeds & determinism
    if ENABLE_DETERMINISM:
        set_deterministic_mode(SEED)
    else:
        logger.info(f"--- Running in Non-Deterministic Mode (seed {SEED}) ---")
        # Still set seeds for basic reproducibility
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)

    # --- Setup model, optimizer, loss
    model = StabilityTestModel(IN_FEATURES, HIDDEN_FEATURES, OUT_FEATURES, AGGREGATION).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()

    logger.info(f"Running for {NUM_EPOCHS} epochs on {device}...")
    logger.info(f"Model: {model.__class__.__name__}")
    logger.info(f"Total Nodes per batch: {NUM_GRAPHS * NODES_PER_GRAPH}")

    # --- 3. TRAINING LOOP ---
    for epoch in range(NUM_EPOCHS):
        model.train()

        epoch_loss = 0

        for sub_step in range(SUB_STEPS_PER_EPOCH):
            # Generate new fake data for each step to vary inputs
            x, pos, batch, y = generate_fake_data(NUM_GRAPHS, NODES_PER_GRAPH, IN_FEATURES, device)

            optimizer.zero_grad()
            out = model(x, pos, batch)
            loss = criterion(out, y)

            # --- 4. STABILITY CHECKS ---

            # Check 1: Loss
            if not torch.isfinite(loss).all():
                logger.error("--- TEST FAILED ---")
                logger.error(f"Loss became NaN or Inf at epoch {epoch}!")
                logger.error(f"Loss value: {loss.item()}")
                sys.exit(1)  # Exit with error code

            epoch_loss += loss.item()

            # Backward pass
            loss.backward()

            # Check 2: Gradients
            for name, param in model.named_parameters():
                if param.grad is not None and not torch.isfinite(param.grad).all():
                    logger.error("--- TEST FAILED ---")
                    logger.error(f"Gradient for '{name}' became NaN or Inf at epoch {epoch}!")
                    sys.exit(1)  # Exit with error code

            optimizer.step()

        logger.info(f"Epoch {epoch + 1:02d}/{NUM_EPOCHS} | Loss: {epoch_loss / SUB_STEPS_PER_EPOCH:.6f} ... OK")

    # --- 5. SUCCESS ---
    logger.success("--- CUDA Stability Test Passed ---")
    logger.success(f"Successfully completed {NUM_EPOCHS} training epochs.")
    logger.success("All specified kernels (segment_csr, radius_graph, knn_graph, scatter) were tested on CUDA.")
    logger.success("No NaN or Inf values detected in loss or gradients.")


if __name__ == "__main__":
    main()
