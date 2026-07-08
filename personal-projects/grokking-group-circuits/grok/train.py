"""Full-batch training loop for a single group-operation task, with periodic
checkpointing of train/test accuracy and the embedding table's spectral
concentration score (see grok/spectral.py)."""
import numpy as np

from .groups import group_task_pairs
from .model import init_params, forward, softmax_cross_entropy, loss_and_grads
from .optim import AdamW
from .spectral import basis_blocks, block_energies, concentration_score, alignment_score


def train_group_task(group, emb_dim=32, hidden_dim=128, steps=3000, lr=1e-3,
                      weight_decay=1.0, train_frac=0.5, seed=0, checkpoint_every=25,
                      final_n_shuffles=300):
    rng = np.random.default_rng(seed)
    a_idx, b_idx, labels = group_task_pairs(group)
    n = len(labels)
    perm = rng.permutation(n)
    n_train = int(round(n * train_frac))
    train_idx = perm[:n_train]
    test_idx = perm[n_train:]

    params = init_params(group.order, emb_dim, hidden_dim, rng)
    opt = AdamW(params, lr=lr, betas=(0.9, 0.98), weight_decay=weight_decay)
    blocks = basis_blocks(group)

    history = {"step": [], "train_loss": [], "train_acc": [],
               "test_loss": [], "test_acc": [], "concentration": []}

    checkpoint_steps = set(range(checkpoint_every, steps + 1, checkpoint_every))
    checkpoint_steps.add(1)
    checkpoint_steps.add(steps)

    initial_concentration = concentration_score(block_energies(params["W_E"], blocks))

    for step in range(1, steps + 1):
        loss, acc, grads = loss_and_grads(
            params, a_idx[train_idx], b_idx[train_idx], labels[train_idx])
        opt.step(params, grads)

        if step in checkpoint_steps:
            test_logits, _ = forward(params, a_idx[test_idx], b_idx[test_idx])
            test_loss, _, test_acc = softmax_cross_entropy(test_logits, labels[test_idx])
            conc = concentration_score(block_energies(params["W_E"], blocks))
            history["step"].append(step)
            history["train_loss"].append(loss)
            history["train_acc"].append(acc)
            history["test_loss"].append(float(test_loss))
            history["test_acc"].append(test_acc)
            history["concentration"].append(conc)

    final_alignment = alignment_score(params["W_E"], blocks, n_shuffles=final_n_shuffles, rng=rng)

    return {
        "history": history,
        "initial_concentration": initial_concentration,
        "final_alignment": final_alignment,
        "params": params,
        "blocks": blocks,
        "train_idx": train_idx,
        "test_idx": test_idx,
    }


def step_to_reach(history, key, threshold):
    """First checkpoint step at which `history[key]` reaches `threshold`,
    or None if it never does."""
    for step, value in zip(history["step"], history[key]):
        if value >= threshold:
            return step
    return None
