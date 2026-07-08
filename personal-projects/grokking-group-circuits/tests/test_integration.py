import numpy as np

from grok.groups import CyclicGroup, DihedralGroup
from grok.train import train_group_task, step_to_reach
from grok.spectral import basis_blocks, alignment_score


def test_tiny_training_run_end_to_end_reduces_loss_and_improves_accuracy():
    """Not a full grokking check (too slow for a unit test) -- just verifies
    the whole pipeline (dataset, model, optimizer, checkpointing, spectral
    scoring) runs to completion and the loss goes down / train accuracy goes
    up on a tiny toy group."""
    group = CyclicGroup(5)
    result = train_group_task(
        group, emb_dim=6, hidden_dim=16, steps=200, lr=1e-2,
        weight_decay=0.0, train_frac=0.8, seed=0, checkpoint_every=20,
        final_n_shuffles=20,
    )
    history = result["history"]
    assert history["train_loss"][-1] < history["train_loss"][0]
    assert history["train_acc"][-1] > history["train_acc"][0]
    assert 0.0 <= result["final_alignment"]["score"] <= 1.0


def test_full_grokking_run_on_small_cyclic_group_generalizes_with_delay():
    """A real (small-scale) grokking run: train accuracy should saturate long
    before test accuracy does, and the final embedding table should show
    significant structure in the group's Fourier basis. This is the actual
    phenomenon the whole project studies, so it's worth one slower
    end-to-end check (~10-15s) rather than only mocking the pieces."""
    group = CyclicGroup(47)
    result = train_group_task(
        group, emb_dim=16, hidden_dim=64, steps=4000, lr=2e-3,
        weight_decay=1.0, train_frac=0.5, seed=0, checkpoint_every=100,
        final_n_shuffles=50,
    )
    history = result["history"]
    train95 = step_to_reach(history, "train_acc", 0.95)
    test95 = step_to_reach(history, "test_acc", 0.95)
    assert train95 is not None
    assert test95 is not None
    assert test95 > train95 * 3  # a real generalization delay, not immediate fit
    assert result["final_alignment"]["z"] > 3.0  # significantly structured vs shuffle null


def test_train_test_split_is_disjoint_and_covers_all_pairs():
    group = DihedralGroup(9)
    result = train_group_task(
        group, emb_dim=4, hidden_dim=8, steps=1, lr=1e-3,
        weight_decay=0.0, train_frac=0.6, seed=1, checkpoint_every=1,
        final_n_shuffles=5,
    )
    train_idx, test_idx = set(result["train_idx"].tolist()), set(result["test_idx"].tolist())
    assert train_idx.isdisjoint(test_idx)
    assert train_idx | test_idx == set(range(group.order * group.order))


def test_different_seeds_give_different_embeddings():
    group = CyclicGroup(7)
    kwargs = dict(emb_dim=4, hidden_dim=8, steps=50, lr=1e-2,
                  weight_decay=0.0, train_frac=0.7, checkpoint_every=25,
                  final_n_shuffles=10)
    r1 = train_group_task(group, seed=0, **kwargs)
    r2 = train_group_task(group, seed=1, **kwargs)
    assert not np.allclose(r1["params"]["W_E"], r2["params"]["W_E"])
