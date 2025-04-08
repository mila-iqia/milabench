import psutil
import socket
import ipaddress
from contextlib import contextmanager

# If true that means we cannot resolve the ip addresses
# so we ignore errors
offline = False


@contextmanager
def enable_offline(enabled):
    global offline
    old = offline

    offline = enabled
    yield


def is_loopback(address: str) -> bool:
    try:
        # Create an IP address object
        ip = ipaddress.ip_address(address)
        # Check if the address is a loopback address
        return ip.is_loopback
    except ValueError:
        # If the address is invalid, return False
        return False


def local_ips() -> set[str]:
    interfaces = psutil.net_if_addrs()
    ip_addresses = []

    for _, addresses in interfaces.items():
        for addr in addresses:
            if addr.family.name == 'AF_INET':  # IPv4
                ip_addresses.append(addr.address)

            elif addr.family.name == 'AF_INET6':  # IPv6
                ip_addresses.append(addr.address)

    return set(ip_addresses)


def gethostbyaddr(addr):
    if offline:
        return addr, [addr]

    try:
        hostname, _, iplist = socket.gethostbyaddr(addr)

        return hostname, iplist
    
    except socket.herror:
        pass
    except socket.gaierror:
        pass
    
    print("Could not resolve address with DNS")
    print("Use IP in your node configuration")
    # This happens if we cannot do a DNS lookup for some reason
    return addr, [addr]


def resolve_ip(ip):
    local = local_ips()

    # we are running code on `cn-l092`
    # gethostbyaddr(172.16.9.192)  cn-l092.server.mila.quebec ['172.16.9.192']
    # gethostbyaddr(cn-l092     )  cn-l092                    ['127.0.1.1']      <= 
    # gethostbyaddr(cn-d003     )  cn-d003.server.mila.quebec ['172.16.8.75']
    # gethostbyaddr(172.16.8.75 )  cn-d003.server.mila.quebec ['172.16.8.75']

    hostname, iplist = gethostbyaddr(ip)
    real_ip = ip

    if len(iplist) == 1 and not is_loopback(iplist[0]):
        real_ip = iplist[0]

    # we need this because
    #
    #   hostname, _, ['127.0.1.1'] = socket.gethostbyaddr("cn-l092")
    #
    # and 127.0.1.1 is not included in our list of IPs
    #
    for ip_entry in iplist:
        if is_loopback(ip_entry):
            return hostname, real_ip, True
    
    if local.intersection(iplist):
        return hostname, real_ip, True

    if hostname == socket.gethostname():
        return hostname, real_ip, True

    return hostname, real_ip, False


def normalize_local(node):
    # Local node usually get stuck resolving local loopback
    # depending on how it was configured
    # this fetch the outbound ip and hostname
    assert node["local"] is True

    # 
    hostname, ip, local = resolve_ip(socket.getfqdn())

    node["hostname"] = hostname
    if '.' not in node["ip"]:
        node["ip"] = ip
    
    # assert local is True


def resolve_node_address(node):
    hostname, ip, local = resolve_ip(node["ip"])

    node["hostname"] = hostname
    node["ip"] = ip
    node["local"] = local

    if local:
        try:
            # `gethostbyaddr` returns `cn-d003` but we want `cn-d003.server.mila.quebec`
            # else torchrun does not recognize the main node
            node["hostname"] = socket.getfqdn()
            
            normalize_local(node)
        except Exception:
            print("Skipped local normalization")

    if offline:
        node["hostname"] = ip

    return local


def resolve_addresses(nodes):
    # This normalize the node ip/hostname
    # for convenience we support a range of values in the IP field
    # we use DNS lookup to resolve the IP/hostname and normalize the fields
    #
    # If DNS is not available then we just leave things as is
    # we also try to find the node we are currently running code on
    # we do that by simply checking all the available IP on this node
    # and check which node has that IP
    self = None
    for node in nodes:
        if resolve_node_address(node):
            self = node

    return self



