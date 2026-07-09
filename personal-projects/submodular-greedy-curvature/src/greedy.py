"""The standard greedy algorithm for cardinality-constrained monotone
submodular maximization (Nemhauser, Wolsey & Fisher, 1978)."""


def greedy(f, k):
    """Return (selected_order, value_trace) for max_{|S|<=k} f(S).

    selected_order: list of element indices in the order greedy added them.
    value_trace: f(S) after each addition (same length as selected_order).
    """
    if k < 0:
        raise ValueError("k must be non-negative")
    k = min(k, f.n)

    selected = []
    selected_set = set()
    value_trace = []
    running_value = 0.0

    for _ in range(k):
        best_j, best_gain = None, -1.0
        for j in range(f.n):
            if j in selected_set:
                continue
            gain = f.marginal_gain(j, selected_set)
            if gain > best_gain:
                best_gain, best_j = gain, j
        selected.append(best_j)
        selected_set.add(best_j)
        running_value += best_gain
        value_trace.append(running_value)

    return selected, value_trace
