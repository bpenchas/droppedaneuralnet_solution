import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import random
import time
import hashlib

# Load all pieces
def load_pieces(pieces_dir="historical_data_and_pieces/pieces"):
    pieces = {}
    for i in range(97):
        pieces[i] = torch.load(f'{pieces_dir}/piece_{i}.pth', map_location='cpu', weights_only=True)
    return pieces

# Load historical data
def load_data(data_path="historical_data_and_pieces/historical_data.csv"):
    df = pd.read_csv(data_path)
    X = df[[f'measurement_{i}' for i in range(48)]].values
    pred = df['pred'].values
    return torch.tensor(X, dtype=torch.float32), torch.tensor(pred, dtype=torch.float32)

# Part 1: Identify LastLayer (shape 1, 48)
def identify_layers(pieces):
    last_layer_idx = None
    inp_layers = []  # (96, 48) - Linear(48, 96) - inp in Block
    out_layers = []  # (48, 96) - Linear(96, 48) - out in Block

    for idx, state_dict in pieces.items():
        weight_shape = tuple(state_dict['weight'].shape)
        if weight_shape == (1, 48):
            last_layer_idx = idx
        elif weight_shape == (96, 48):
            inp_layers.append(idx)
        elif weight_shape == (48, 96):
            out_layers.append(idx)

    return last_layer_idx, inp_layers, out_layers


# Part 2: Pair inp and out layers using per-neuron cosine similarity
def cosine_similarity_per_neuron(inp_weight, out_weight):
    """
    inp_weight: (96, 48) - each row is a neuron's input weights
    out_weight: (48, 96) - each column is a neuron's output weights

    For matching pairs, the input weights of neuron i (inp_weight[i])
    should be related to the output weights of neuron i (out_weight[:, i])
    """
    # Transpose out_weight to get (96, 48) where each row is a neuron's output weights
    out_weight_t = out_weight.T  # (96, 48)

    # Compute cosine similarity between corresponding neurons
    # inp_weight[i] vs out_weight_t[i] for each neuron i
    inp_norm = inp_weight / (inp_weight.norm(dim=1, keepdim=True) + 1e-8)
    out_norm = out_weight_t / (out_weight_t.norm(dim=1, keepdim=True) + 1e-8)

    # Per-neuron cosine similarity
    cos_sim = (inp_norm * out_norm).sum(dim=1)  # (96,)

    # Use absolute value since weights can be flipped
    return cos_sim.abs().mean().item()


def pair_inp_out_layers(pieces, inp_layers, out_layers):
    """Pair each inp layer with its matching out layer using cosine similarity."""
    pairs = []
    remaining_out = set(out_layers)

    for inp_idx in inp_layers:
        inp_weight = pieces[inp_idx]['weight']

        best_out_idx = None
        best_similarity = -1

        for out_idx in remaining_out:
            out_weight = pieces[out_idx]['weight']
            similarity = cosine_similarity_per_neuron(inp_weight, out_weight)

            if similarity > best_similarity:
                best_similarity = similarity
                best_out_idx = out_idx

        pairs.append((inp_idx, best_out_idx))
        remaining_out.remove(best_out_idx)

    return pairs


# Block and LastLayer classes
class Block(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.inp = nn.Linear(in_dim, hidden_dim)
        self.activation = nn.ReLU()
        self.out = nn.Linear(hidden_dim, in_dim)

    def forward(self, x):
        residual = x
        x = self.inp(x)
        x = self.activation(x)
        x = self.out(x)
        return residual + x


class LastLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.layer = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.layer(x)


def build_model(pieces, block_pairs, last_layer_idx):
    """Build a model from the given block ordering."""
    blocks = []
    for inp_idx, out_idx in block_pairs:
        block = Block(48, 96)
        block.inp.load_state_dict(pieces[inp_idx])
        block.out.load_state_dict(pieces[out_idx])
        blocks.append(block)

    last_layer = LastLayer(48, 1)
    last_layer.layer.load_state_dict(pieces[last_layer_idx])

    model = nn.Sequential(*blocks, last_layer)
    return model


def evaluate_ordering(pieces, block_pairs, last_layer_idx, X, pred):
    """Evaluate MSE between model output and predictions."""
    model = build_model(pieces, block_pairs, last_layer_idx)
    model.eval()

    with torch.no_grad():
        output = model(X).squeeze()
        mse = ((output - pred) ** 2).mean().item()

    return mse


# Part 3: Simulated Annealing
def simulated_annealing(pieces, block_pairs, last_layer_idx, X, pred,
                        T_start=0.005, T_end=1e-8, max_iterations=100000, log_interval=1000):
    """Find optimal block ordering using simulated annealing."""
    current_pairs = list(block_pairs)
    current_mse = evaluate_ordering(pieces, current_pairs, last_layer_idx, X, pred)

    best_pairs = list(current_pairs)
    best_mse = current_mse

    n_blocks = len(current_pairs)

    # Calculate temperature decay rate
    decay_rate = (T_end / T_start) ** (1 / max_iterations)
    T = T_start

    # Logging history for charting
    history = []
    start_time = time.time()

    for iteration in range(max_iterations):
        # Generate neighbor by swapping two blocks
        new_pairs = list(current_pairs)
        i, j = random.sample(range(n_blocks), 2)
        new_pairs[i], new_pairs[j] = new_pairs[j], new_pairs[i]

        new_mse = evaluate_ordering(pieces, new_pairs, last_layer_idx, X, pred)

        # Accept or reject
        delta = new_mse - current_mse
        if delta < 0 or random.random() < np.exp(-delta / T):
            current_pairs = new_pairs
            current_mse = new_mse

            if current_mse < best_mse:
                best_pairs = list(current_pairs)
                best_mse = current_mse

                # Early termination if perfect solution found
                if best_mse == 0:
                    elapsed = time.time() - start_time
                    history.append({
                        'step': iteration,
                        'best_mse': best_mse,
                        'current_mse': current_mse,
                        'temperature': T,
                        'time': elapsed
                    })
                    print(f"Perfect solution found at iteration {iteration}!")
                    return best_pairs, best_mse, history

        # Decay temperature
        T *= decay_rate

        if iteration % log_interval == 0:
            elapsed = time.time() - start_time
            history.append({
                'step': iteration,
                'best_mse': best_mse,
                'current_mse': current_mse,
                'temperature': T,
                'time': elapsed
            })
            print(f"Iteration {iteration}: T={T:.2e}, current_mse={current_mse:.2e}, best_mse={best_mse:.2e}, time={elapsed:.1f}s")

    # Log final state
    elapsed = time.time() - start_time
    history.append({
        'step': max_iterations,
        'best_mse': best_mse,
        'current_mse': current_mse,
        'temperature': T,
        'time': elapsed
    })

    print(f"Final best MSE: {best_mse:.6f}")
    return best_pairs, best_mse, history


def pairs_to_permutation(block_pairs, last_layer_idx):
    """Convert block pairs to permutation format.

    The solution is a permutation where position i contains the piece index
    that should be applied at position i.

    Structure: [inp_0, out_0, inp_1, out_1, ..., inp_47, out_47, last_layer]
    """
    permutation = []
    for inp_idx, out_idx in block_pairs:
        permutation.append(inp_idx)
        permutation.append(out_idx)
    permutation.append(last_layer_idx)
    return permutation


def check_solution(permutation_str):
    """Check if the provided permutation matches the expected hash."""
    try:
        perm = [int(x.strip()) for x in permutation_str.split(',')]
        if len(perm) != 97 or set(perm) != set(range(97)):
            return f"Expected 97 numbers (indices 0-96), got {len(perm)}"

        # Create canonical string representation for hashing
        canonical = ','.join(str(x) for x in perm)
        solution_hash = hashlib.sha256(canonical.encode()).hexdigest()

        expected_hash = "093be1cf2d24094db903cbc3e8d33d306ebca49c6accaa264e44b0b675e7d9c4"

        if solution_hash == expected_hash:
            return "You've reconstructed the model!"
        else:
            return f"Incorrect. Your hash: {solution_hash[:16]}..."
    except ValueError as e:
        return f"Parse error: {e}. Please provide a list of comma separated integers"


def main():
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    print("Loading pieces and data...")
    pieces = load_pieces()
    X_full, pred_full = load_data()

    # Use subset for faster simulated annealing
    n_samples = 200
    X = X_full[:n_samples]
    pred = pred_full[:n_samples]
    print(f"Using {n_samples} samples for optimization (out of {len(X_full)})")

    print("Identifying layer types...")
    last_layer_idx, inp_layers, out_layers = identify_layers(pieces)
    print(f"LastLayer: piece_{last_layer_idx}")
    print(f"Input layers: {len(inp_layers)} pieces")
    print(f"Output layers: {len(out_layers)} pieces")

    print("\nPairing inp and out layers using cosine similarity...")
    block_pairs = pair_inp_out_layers(pieces, inp_layers, out_layers)

    # Show initial MSE before ordering optimization
    initial_mse = evaluate_ordering(pieces, block_pairs, last_layer_idx, X, pred)
    print(f"Initial MSE with random block ordering: {initial_mse:.6f}")

    print("\nRunning simulated annealing to find optimal block ordering...")
    best_pairs, best_mse, history = simulated_annealing(
        pieces, block_pairs, last_layer_idx, X, pred,
        T_start=0.005, T_end=1e-8, max_iterations=100000
    )

    # Save history to CSV for charting
    history_df = pd.DataFrame(history)
    history_df.to_csv('annealing_history.csv', index=False)
    print("\nSaved annealing history to annealing_history.csv")

    # Verify on full dataset
    full_mse = evaluate_ordering(pieces, best_pairs, last_layer_idx, X_full, pred_full)
    print(f"MSE on full dataset: {full_mse:.6f}")

    permutation = pairs_to_permutation(best_pairs, last_layer_idx)

    print("\nFinal permutation (piece index at each position 0-96):")
    print(permutation)

    # Verify solution hash
    permutation_str = ','.join(str(x) for x in permutation)
    result = check_solution(permutation_str)
    print(f"\nSolution verification: {result}")

    return permutation, history


if __name__ == "__main__":
    main()
