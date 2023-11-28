from milabench.utils import enumerate_rank, select_nodes


def test_enumerate_rank():
    nodes = [
        {"main": False},
        {"main": False},
        {"main": True},
        {"main": False},
    ]
    ranks = [r for r, _ in enumerate_rank(nodes)]

    assert ranks == [1, 2, 0, 3]


def test_select_nodes():
    nodes = [
        {"main": False},
        {"main": False},
        {"main": True},
        {"main": False},
    ]

    selected = select_nodes(nodes, 3)
    assert selected == [{"main": True}, {"main": False}, {"main": False}]
