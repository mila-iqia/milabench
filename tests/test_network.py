from milabench.network import resolve_addresses, enable_offline

from copy import deepcopy
import pytest


# test that it works without DNS
# podman run --rm -it --dns 0.0.0.0 -v "$(pwd):/mnt"  python-with-psutil bash

cases = [
    {
        "ip": "172.16.9.192"
    },
    {
        "ip": "cn-l092"
    },
    {
        "ip": "cn-d003"
    },
    {
        "ip": "cn-l092.server.mila.quebec"
    },
    {
        "ip": "cn-d003.server.mila.quebec"
    },
    {
        "ip": "172.16.8.75"
    }
]


@pytest.mark.skip(reason="those hostnames mean nothing in the CI")
def test_network():
    nodes = deepcopy(cases)
    print(resolve_addresses(nodes))
    print()
    for n in nodes:
        print(n)


@pytest.mark.skip(reason="those hostnames mean nothing in the CI")
def test_no_dns_network():
    with enable_offline():
        nodes = deepcopy(cases)
        print(resolve_addresses(nodes))
        print()
        for n in nodes:
            print(n)


def check_dns():
    nodes = deepcopy(cases)
    print(resolve_addresses(nodes))
    print()
    for n in nodes:
        print(n)

    print("===")
    with enable_offline():
        nodes = deepcopy(cases)
        print(resolve_addresses(nodes))
        print()
        for n in nodes:
            print(n)


if __name__ == "__main__":
    check_dns()
