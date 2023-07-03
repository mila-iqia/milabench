from milabench.slurm import expand_node_list


def test_slurm_node_expand_0():
    assert expand_node_list("node1,cn-d[003-004,006],node5,kep-d[03-04,06]") == [
        "node1",
        "cn-d003",
        "cn-d004",
        "cn-d006",
        "node5",
        "kep-d03",
        "kep-d04",
        "kep-d06",
    ]


def test_slurm_node_expand_1():
    assert expand_node_list("cn-d[003-008]") == [
        "cn-d003",
        "cn-d004",
        "cn-d005",
        "cn-d006",
        "cn-d007",
        "cn-d008",
    ]


def test_slurm_node_expand_2():
    assert expand_node_list("cn-d[003,005,007]") == ["cn-d003", "cn-d005", "cn-d007"]
