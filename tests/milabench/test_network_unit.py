"""Tests for milabench.network module."""

import socket
from collections import namedtuple
from unittest.mock import patch, MagicMock

import pytest

from milabench import network
from milabench.network import (
    enable_offline,
    is_loopback,
    local_ips,
    gethostbyaddr,
    resolve_ip,
    normalize_local,
    resolve_node_address,
    resolve_addresses,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _reset_offline():
    """Ensure the module-level ``offline`` flag is restored after each test."""
    original = network.offline
    yield
    network.offline = original


@pytest.fixture()
def fake_interfaces():
    """Return a deterministic psutil-style interface dict."""
    Addr = namedtuple("Addr", ["family", "address", "netmask", "broadcast", "ptp"])

    class _Family:
        def __init__(self, name):
            self.name = name

    return {
        "eth0": [
            Addr(family=_Family("AF_INET"), address="192.168.1.10", netmask=None, broadcast=None, ptp=None),
            Addr(family=_Family("AF_INET6"), address="fe80::1", netmask=None, broadcast=None, ptp=None),
        ],
        "lo": [
            Addr(family=_Family("AF_INET"), address="127.0.0.1", netmask=None, broadcast=None, ptp=None),
        ],
        "wlan0": [
            Addr(family=_Family("AF_LINK"), address="aa:bb:cc:dd:ee:ff", netmask=None, broadcast=None, ptp=None),
        ],
    }


# ---------------------------------------------------------------------------
# enable_offline
# ---------------------------------------------------------------------------

class TestEnableOffline:
    def test_sets_value_inside_context(self):
        network.offline = False
        with enable_offline(True):
            assert network.offline is True

    def test_noop_when_already_matching(self):
        network.offline = True
        with enable_offline(True):
            assert network.offline is True

    def test_switches_to_false(self):
        network.offline = True
        with enable_offline(False):
            assert network.offline is False


# ---------------------------------------------------------------------------
# is_loopback
# ---------------------------------------------------------------------------

class TestIsLoopback:
    def test_ipv4_loopback(self):
        assert is_loopback("127.0.0.1") is True

    def test_ipv4_loopback_alternate(self):
        assert is_loopback("127.0.1.1") is True

    def test_ipv6_loopback(self):
        assert is_loopback("::1") is True

    def test_non_loopback_ipv4(self):
        assert is_loopback("192.168.1.1") is False

    def test_non_loopback_ipv6(self):
        assert is_loopback("fe80::1") is False

    def test_invalid_address_returns_false(self):
        """Covers lines 26-28: ValueError branch."""
        assert is_loopback("not-an-ip") is False

    def test_empty_string_returns_false(self):
        assert is_loopback("") is False

    def test_hostname_returns_false(self):
        assert is_loopback("example.com") is False


# ---------------------------------------------------------------------------
# local_ips
# ---------------------------------------------------------------------------

class TestLocalIps:
    def test_returns_ipv4_and_ipv6(self, fake_interfaces):
        with patch("milabench.network.psutil.net_if_addrs", return_value=fake_interfaces):
            result = local_ips()

        assert "192.168.1.10" in result
        assert "fe80::1" in result
        assert "127.0.0.1" in result

    def test_excludes_non_ip_families(self, fake_interfaces):
        with patch("milabench.network.psutil.net_if_addrs", return_value=fake_interfaces):
            result = local_ips()

        assert "aa:bb:cc:dd:ee:ff" not in result

    def test_returns_set(self, fake_interfaces):
        with patch("milabench.network.psutil.net_if_addrs", return_value=fake_interfaces):
            result = local_ips()

        assert isinstance(result, set)

    def test_empty_interfaces(self):
        with patch("milabench.network.psutil.net_if_addrs", return_value={}):
            result = local_ips()

        assert result == set()


# ---------------------------------------------------------------------------
# gethostbyaddr
# ---------------------------------------------------------------------------

class TestGethostbyaddr:
    def test_offline_mode_returns_addr_directly(self):
        with enable_offline(True):
            hostname, iplist = gethostbyaddr("10.0.0.1")

        assert hostname == "10.0.0.1"
        assert iplist == ["10.0.0.1"]

    def test_successful_resolution(self):
        with patch("milabench.network.socket.gethostbyaddr") as mock_resolve:
            mock_resolve.return_value = ("host.example.com", [], ["10.0.0.1"])
            hostname, iplist = gethostbyaddr("10.0.0.1")

        assert hostname == "host.example.com"
        assert iplist == ["10.0.0.1"]

    def test_herror_falls_back(self, capsys):
        """Covers lines 55-56: socket.herror branch."""
        with patch("milabench.network.socket.gethostbyaddr") as mock_resolve:
            mock_resolve.side_effect = socket.herror("Host not found")
            hostname, iplist = gethostbyaddr("10.0.0.99")

        assert hostname == "10.0.0.99"
        assert iplist == ["10.0.0.99"]
        captured = capsys.readouterr()
        assert "Could not resolve address with DNS" in captured.out

    def test_gaierror_falls_back(self, capsys):
        """Covers lines 57-58: socket.gaierror branch."""
        with patch("milabench.network.socket.gethostbyaddr") as mock_resolve:
            mock_resolve.side_effect = socket.gaierror("Name resolution failed")
            hostname, iplist = gethostbyaddr("bad-host")

        assert hostname == "bad-host"
        assert iplist == ["bad-host"]
        captured = capsys.readouterr()
        assert "Use IP in your node configuration" in captured.out


# ---------------------------------------------------------------------------
# resolve_ip
# ---------------------------------------------------------------------------

class TestResolveIp:
    def test_remote_non_loopback_ip(self):
        with patch("milabench.network.local_ips", return_value={"192.168.1.10"}), \
             patch("milabench.network.gethostbyaddr", return_value=("remote.host", ["10.0.0.5"])):
            hostname, ip, local = resolve_ip("10.0.0.5")

        assert hostname == "remote.host"
        assert ip == "10.0.0.5"
        assert local is False

    def test_loopback_in_iplist_marks_local(self):
        """When gethostbyaddr returns a loopback, the node is local."""
        with patch("milabench.network.local_ips", return_value={"192.168.1.10"}), \
             patch("milabench.network.gethostbyaddr", return_value=("myhost", ["127.0.1.1"])):
            hostname, ip, local = resolve_ip("myhost")

        assert hostname == "myhost"
        assert ip == "myhost"  # loopback, so real_ip stays as original
        assert local is True

    def test_local_ip_intersection_marks_local(self):
        """Covers line 92: local.intersection(iplist) branch."""
        with patch("milabench.network.local_ips", return_value={"192.168.1.10"}), \
             patch("milabench.network.gethostbyaddr", return_value=("myhost", ["192.168.1.10"])):
            hostname, ip, local = resolve_ip("192.168.1.10")

        assert hostname == "myhost"
        assert ip == "192.168.1.10"
        assert local is True

    def test_hostname_matches_gethostname_marks_local(self):
        """Covers line 95: hostname == socket.gethostname() branch."""
        with patch("milabench.network.local_ips", return_value={"10.0.0.1"}), \
             patch("milabench.network.gethostbyaddr", return_value=("this-host", ["10.0.0.50"])), \
             patch("milabench.network.socket.gethostname", return_value="this-host"):
            hostname, ip, local = resolve_ip("10.0.0.50")

        assert hostname == "this-host"
        assert ip == "10.0.0.50"
        assert local is True

    def test_multiple_ips_in_iplist_keeps_original(self):
        """When iplist has more than one entry, real_ip stays as the input."""
        with patch("milabench.network.local_ips", return_value=set()), \
             patch("milabench.network.gethostbyaddr", return_value=("host", ["10.0.0.1", "10.0.0.2"])), \
             patch("milabench.network.socket.gethostname", return_value="other"):
            hostname, ip, local = resolve_ip("original-ip")

        assert ip == "original-ip"
        assert local is False

    def test_single_loopback_keeps_original_ip(self):
        """Single loopback in iplist: real_ip stays as original (loopback is skipped)."""
        with patch("milabench.network.local_ips", return_value=set()), \
             patch("milabench.network.gethostbyaddr", return_value=("host", ["127.0.0.1"])):
            hostname, ip, local = resolve_ip("some-host")

        assert ip == "some-host"
        assert local is True


# ---------------------------------------------------------------------------
# normalize_local
# ---------------------------------------------------------------------------

class TestNormalizeLocal:
    def test_updates_hostname_and_ip(self):
        node = {"local": True, "ip": "myhost", "hostname": ""}
        with patch("milabench.network.resolve_ip", return_value=("fqdn.host.com", "10.0.0.1", True)), \
             patch("milabench.network.socket.getfqdn", return_value="fqdn.host.com"):
            normalize_local(node)

        assert node["hostname"] == "fqdn.host.com"
        assert node["ip"] == "10.0.0.1"

    def test_does_not_overwrite_dotted_ip(self):
        """Covers line 111: ip already contains a dot, skip overwrite."""
        node = {"local": True, "ip": "10.0.0.5", "hostname": ""}
        with patch("milabench.network.resolve_ip", return_value=("fqdn.host.com", "10.0.0.99", True)), \
             patch("milabench.network.socket.getfqdn", return_value="fqdn.host.com"):
            normalize_local(node)

        assert node["hostname"] == "fqdn.host.com"
        assert node["ip"] == "10.0.0.5"  # not overwritten

    def test_asserts_on_non_local_node(self):
        node = {"local": False, "ip": "10.0.0.1", "hostname": ""}
        with pytest.raises(AssertionError):
            normalize_local(node)


# ---------------------------------------------------------------------------
# resolve_node_address
# ---------------------------------------------------------------------------

class TestResolveNodeAddress:
    def test_remote_node(self):
        node = {"ip": "10.0.0.5", "hostname": "", "local": False}
        with patch("milabench.network.resolve_ip", return_value=("remote.host", "10.0.0.5", False)):
            result = resolve_node_address(node)

        assert result is False
        assert node["hostname"] == "remote.host"
        assert node["ip"] == "10.0.0.5"
        assert node["local"] is False

    def test_local_node_normalizes(self):
        node = {"ip": "myhost", "hostname": "", "local": False}
        with patch("milabench.network.resolve_ip", return_value=("myhost", "10.0.0.1", True)), \
             patch("milabench.network.socket.getfqdn", return_value="myhost.fqdn.com"), \
             patch("milabench.network.normalize_local") as mock_norm:
            result = resolve_node_address(node)

        assert result is True
        assert node["hostname"] == "myhost.fqdn.com"
        mock_norm.assert_called_once_with(node)

    def test_local_node_normalize_exception_caught(self, capsys):
        """Covers lines 130-131: exception during normalize_local."""
        node = {"ip": "myhost", "hostname": "", "local": False}
        with patch("milabench.network.resolve_ip", return_value=("myhost", "10.0.0.1", True)), \
             patch("milabench.network.socket.getfqdn", side_effect=Exception("DNS failure")):
            result = resolve_node_address(node)

        assert result is True
        captured = capsys.readouterr()
        assert "Skipped local normalization" in captured.out

    def test_offline_overrides_hostname(self):
        node = {"ip": "10.0.0.5", "hostname": "", "local": False}
        with enable_offline(True), \
             patch("milabench.network.resolve_ip", return_value=("remote.host", "10.0.0.5", False)):
            resolve_node_address(node)

        assert node["hostname"] == "10.0.0.5"

    def test_offline_local_node(self):
        node = {"ip": "myhost", "hostname": "", "local": False}
        with enable_offline(True), \
             patch("milabench.network.resolve_ip", return_value=("myhost", "10.0.0.1", True)), \
             patch("milabench.network.socket.getfqdn", return_value="myhost.fqdn.com"), \
             patch("milabench.network.normalize_local"):
            resolve_node_address(node)

        assert node["hostname"] == "10.0.0.1"


# ---------------------------------------------------------------------------
# resolve_addresses
# ---------------------------------------------------------------------------

class TestResolveAddresses:
    def test_returns_local_node(self):
        nodes = [
            {"ip": "10.0.0.1", "hostname": "", "local": False},
            {"ip": "10.0.0.2", "hostname": "", "local": False},
        ]

        def fake_resolve(node):
            node["hostname"] = f"host-{node['ip']}"
            return node["ip"] == "10.0.0.1"

        with patch("milabench.network.resolve_node_address", side_effect=fake_resolve):
            result = resolve_addresses(nodes)

        assert result is nodes[0]

    def test_returns_none_when_no_local(self):
        nodes = [
            {"ip": "10.0.0.1", "hostname": "", "local": False},
        ]
        with patch("milabench.network.resolve_node_address", return_value=False):
            result = resolve_addresses(nodes)

        assert result is None

    def test_empty_node_list(self):
        result = resolve_addresses([])
        assert result is None

    def test_multiple_local_returns_last(self):
        nodes = [
            {"ip": "10.0.0.1"},
            {"ip": "10.0.0.2"},
            {"ip": "10.0.0.3"},
        ]
        with patch("milabench.network.resolve_node_address", return_value=True):
            result = resolve_addresses(nodes)

        assert result is nodes[2]
