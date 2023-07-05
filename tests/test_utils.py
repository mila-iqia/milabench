from milabench.utils import enumerate_rank


def test_enumerate_rank():
    nodes = [
        {"main": False},
        {"main": False},
        {"main": True},
        {"main": False},
        
    ]
    ranks = [r for r, _ in enumerate_rank(nodes)]
    
    assert ranks == [1, 2, 0, 3]
