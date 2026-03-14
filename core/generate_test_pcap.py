"""
generate_test_pcap.py
---------------------
Generates a synthetic .pcap file with realistic test traffic.

Creates packets for:
  - TLS ClientHello (HTTPS) with real SNI values → YouTube, Facebook, etc.
  - Plain HTTP requests with Host headers
  - DNS queries for various domains
  - Generic TCP data flows
  - Suspicious/high-volume traffic (for ML anomaly testing in Phase 3)

Run:
  python3 generate_test_pcap.py
  → Creates test_traffic.pcap
"""

import struct
import socket
import time
import random
import os
import sys

# ── PCAP Write Helpers ────────────────────────────────────────────────────────

def pcap_global_header() -> bytes:
    """Write standard PCAP global header (Ethernet)."""
    return struct.pack("<IHHiIII",
                       0xa1b2c3d4,  # magic
                       2, 4,         # version 2.4
                       0,            # timezone
                       0,            # accuracy
                       65535,        # snaplen
                       1)            # Ethernet

def pcap_packet_header(data: bytes, ts: float) -> bytes:
    ts_sec  = int(ts)
    ts_usec = int((ts - ts_sec) * 1_000_000)
    return struct.pack("<IIII", ts_sec, ts_usec, len(data), len(data))

def eth_header(src_mac: bytes, dst_mac: bytes, ethertype: int = 0x0800) -> bytes:
    return dst_mac + src_mac + struct.pack("!H", ethertype)

def ipv4_header(src_ip: str, dst_ip: str, proto: int,
                payload_len: int, ttl: int = 64, ident: int = None) -> bytes:
    if ident is None:
        ident = random.randint(0, 65535)
    src = socket.inet_aton(src_ip)
    dst = socket.inet_aton(dst_ip)
    total_len = 20 + payload_len
    # Build header without checksum first
    hdr = struct.pack("!BBHHHBBH4s4s",
                      0x45,        # version=4, IHL=5
                      0,           # DSCP/ECN
                      total_len,
                      ident,
                      0x4000,      # DF flag
                      ttl,
                      proto,
                      0,           # checksum placeholder
                      src,
                      dst)
    chk = _checksum(hdr)
    return hdr[:10] + struct.pack("!H", chk) + hdr[12:]

def tcp_header(src_port: int, dst_port: int, seq: int = 0, ack: int = 0,
               flags: int = 0x018, window: int = 65535) -> bytes:
    # flags: 0x018 = PSH+ACK, 0x002 = SYN, 0x010 = ACK
    hdr = struct.pack("!HHIIBBHHH",
                      src_port, dst_port,
                      seq, ack,
                      0x50,     # data offset = 5 (20 bytes), reserved
                      flags,
                      window,
                      0,        # checksum
                      0)        # urgent
    return hdr

def udp_header(src_port: int, dst_port: int, payload_len: int) -> bytes:
    return struct.pack("!HHHH",
                       src_port, dst_port,
                       8 + payload_len,
                       0)

def _checksum(data: bytes) -> int:
    if len(data) % 2:
        data += b'\x00'
    total = sum(struct.unpack("!" + "H" * (len(data) // 2), data))
    total = (total >> 16) + (total & 0xFFFF)
    total += (total >> 16)
    return ~total & 0xFFFF

# ── Payload Builders ──────────────────────────────────────────────────────────

# Random MAC addresses
CLIENT_MAC = bytes([0x00, 0x11, 0x22, 0x33, 0x44, 0x55])
SERVER_MAC = bytes([0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF])
GATEWAY_MAC = bytes([0x11, 0x22, 0x33, 0x44, 0x55, 0x66])


def build_tls_client_hello(sni: str) -> bytes:
    """
    Build a realistic TLS 1.2 ClientHello with SNI extension.
    This is the packet DPI engines inspect to extract the domain name.
    """
    sni_bytes = sni.encode("ascii")
    sni_len   = len(sni_bytes)

    # SNI extension content
    sni_ext_content = struct.pack("!HBH", sni_len + 3, 0x00, sni_len) + sni_bytes
    sni_ext_data    = struct.pack("!HH", 0x0000, len(sni_ext_content)) + sni_ext_content

    # Supported groups extension
    groups_data = struct.pack("!HH", 0x000A, 8) + struct.pack("!H", 6) + \
                  struct.pack("!HHHH", 0x001D, 0x0017, 0x0018, 0x0019)  # groups

    # Session tickets
    ticket_ext = struct.pack("!HH", 0x0023, 0)

    # Supported versions (TLS 1.3 advertised)
    ver_ext = struct.pack("!HH", 0x002B, 5) + struct.pack("!B", 4) + \
              struct.pack("!HH", 0x0304, 0x0303)

    all_extensions = sni_ext_data + groups_data + ticket_ext + ver_ext
    ext_len = len(all_extensions)

    # Cipher suites (common ones)
    ciphers = struct.pack("!HHHHHH",
                          0xC02B, 0xC02F, 0xC02C, 0xC030,
                          0x009C, 0x009D)
    cipher_len = len(ciphers)

    # ClientHello body
    random_bytes = bytes([random.randint(0, 255) for _ in range(32)])
    body = (struct.pack("!H", 0x0303) +    # client version TLS 1.2
            random_bytes +                   # 32 bytes random
            b'\x00' +                        # session ID length = 0
            struct.pack("!H", cipher_len) +
            ciphers +
            b'\x01\x00' +                    # compression: 1 method, null
            struct.pack("!H", ext_len) +
            all_extensions)

    # Handshake header
    body_len = len(body)
    hs = bytes([0x01]) + struct.pack("!I", body_len)[1:] + body   # type + 3-byte len

    # TLS record
    record = bytes([0x16, 0x03, 0x01]) + struct.pack("!H", len(hs)) + hs
    return record


def build_http_request(host: str, path: str = "/") -> bytes:
    """Build a plain HTTP GET request."""
    req = (f"GET {path} HTTP/1.1\r\n"
           f"Host: {host}\r\n"
           f"User-Agent: Mozilla/5.0 (DPI-Test)\r\n"
           f"Accept: */*\r\n"
           f"Connection: keep-alive\r\n"
           f"\r\n")
    return req.encode("ascii")


def build_dns_query(domain: str, qtype: int = 1) -> bytes:
    """Build a DNS query packet (UDP payload)."""
    txid = random.randint(0, 65535)
    flags = 0x0100   # Standard query, recursion desired
    header = struct.pack("!HHHHHH", txid, flags, 1, 0, 0, 0)

    # Encode QNAME
    labels = b""
    for part in domain.split("."):
        enc = part.encode("ascii")
        labels += bytes([len(enc)]) + enc
    labels += b'\x00'   # Root label

    question = labels + struct.pack("!HH", qtype, 1)   # QTYPE A, QCLASS IN
    return header + question


# ── Packet Assembly ───────────────────────────────────────────────────────────

def make_tcp_packet(src_ip: str, dst_ip: str, src_port: int, dst_port: int,
                    payload: bytes, flags: int = 0x018,
                    seq: int = None, ack: int = None,
                    src_mac: bytes = CLIENT_MAC,
                    dst_mac: bytes = SERVER_MAC) -> bytes:
    if seq is None: seq = random.randint(0, 2**32)
    if ack is None: ack = random.randint(0, 2**32)
    tcp  = tcp_header(src_port, dst_port, seq, ack, flags)
    ip   = ipv4_header(src_ip, dst_ip, 6, len(tcp) + len(payload))
    eth  = eth_header(src_mac, dst_mac)
    return eth + ip + tcp + payload


def make_udp_packet(src_ip: str, dst_ip: str, src_port: int, dst_port: int,
                    payload: bytes,
                    src_mac: bytes = CLIENT_MAC,
                    dst_mac: bytes = SERVER_MAC) -> bytes:
    udp = udp_header(src_port, dst_port, len(payload))
    ip  = ipv4_header(src_ip, dst_ip, 17, len(udp) + len(payload))
    eth = eth_header(src_mac, dst_mac)
    return eth + ip + udp + payload


# ── Scenario Generators ───────────────────────────────────────────────────────

def scenario_https_with_sni(packets: list, ts: float,
                              client: str, server: str,
                              sni: str, sport: int) -> float:
    """Generate a full HTTPS session: SYN → TLS ClientHello → data."""
    server_ip = {
        "www.youtube.com":   "142.250.185.206",
        "www.facebook.com":  "157.240.3.35",
        "www.instagram.com": "157.240.3.174",
        "www.tiktok.com":    "162.159.137.1",
        "www.netflix.com":   "54.237.232.117",
        "github.com":        "140.82.113.3",
        "www.google.com":    "142.250.185.110",
        "discord.com":       "162.159.130.233",
        "www.reddit.com":    "151.101.65.140",
        "open.spotify.com":  "35.186.224.47",
    }.get(sni, "1.2.3.4")

    # SYN
    pkt = make_tcp_packet(client, server_ip, sport, 443, b"", flags=0x002)
    packets.append((ts, pkt)); ts += 0.001

    # SYN-ACK
    pkt = make_tcp_packet(server_ip, client, 443, sport, b"", flags=0x012,
                          src_mac=SERVER_MAC, dst_mac=CLIENT_MAC)
    packets.append((ts, pkt)); ts += 0.001

    # ACK
    pkt = make_tcp_packet(client, server_ip, sport, 443, b"", flags=0x010)
    packets.append((ts, pkt)); ts += 0.001

    # TLS ClientHello (the DPI-interesting packet!)
    hello = build_tls_client_hello(sni)
    pkt = make_tcp_packet(client, server_ip, sport, 443, hello, flags=0x018)
    packets.append((ts, pkt)); ts += 0.002

    # Simulate some encrypted data packets
    for _ in range(random.randint(3, 10)):
        data = bytes([random.randint(0, 255) for _ in range(random.randint(100, 1400))])
        pkt = make_tcp_packet(client, server_ip, sport, 443, data, flags=0x018)
        packets.append((ts, pkt)); ts += random.uniform(0.01, 0.1)

    # FIN-ACK
    pkt = make_tcp_packet(client, server_ip, sport, 443, b"", flags=0x011)
    packets.append((ts, pkt)); ts += 0.001

    return ts


def scenario_http_request(packets: list, ts: float,
                           client: str, server: str, host: str,
                           sport: int) -> float:
    """Plain HTTP request."""
    req = build_http_request(host)
    pkt = make_tcp_packet(client, server, sport, 80, req, flags=0x018)
    packets.append((ts, pkt)); ts += 0.002

    # HTTP 200 OK response
    resp = (b"HTTP/1.1 200 OK\r\n"
            b"Content-Type: text/html\r\n"
            b"Content-Length: 100\r\n\r\n" +
            b"X" * 100)
    pkt = make_tcp_packet(server, client, 80, sport, resp, flags=0x018,
                          src_mac=SERVER_MAC, dst_mac=CLIENT_MAC)
    packets.append((ts, pkt)); ts += 0.005

    return ts


def scenario_dns_queries(packets: list, ts: float, client: str) -> float:
    """DNS lookup sequence."""
    domains = [
        "www.youtube.com", "www.facebook.com", "api.spotify.com",
        "www.reddit.com", "discord.com", "api.github.com",
    ]
    dns_server = "8.8.8.8"
    for domain in domains:
        payload = build_dns_query(domain)
        pkt = make_udp_packet(client, dns_server,
                              random.randint(40000, 60000), 53, payload)
        packets.append((ts, pkt)); ts += random.uniform(0.05, 0.2)
    return ts


def scenario_port_scan(packets: list, ts: float, attacker: str) -> float:
    """Simulate a port scan (many SYN packets to different ports)."""
    target = "192.168.1.100"
    ports = random.sample(range(1, 10000), 50)
    for port in ports:
        pkt = make_tcp_packet(attacker, target,
                              random.randint(40000, 60000), port,
                              b"", flags=0x002)
        packets.append((ts, pkt)); ts += 0.001   # Very fast = suspicious
    return ts


def scenario_high_volume(packets: list, ts: float,
                          client: str, server: str) -> float:
    """High-volume data transfer (potential exfiltration)."""
    sport = random.randint(40000, 60000)
    for _ in range(40):
        data = bytes([random.randint(0, 255) for _ in range(1400)])
        pkt = make_tcp_packet(client, server, sport, 443, data, flags=0x018)
        packets.append((ts, pkt)); ts += 0.001   # Fast burst
    return ts


# ── Main Generator ─────────────────────────────────────────────────────────────

def generate_test_pcap(output_path: str = "test_traffic.pcap") -> int:
    """Generate the test PCAP and return packet count."""
    packets = []    # list of (timestamp, raw_bytes)
    ts = time.time() - 300   # Start 5 minutes ago

    CLIENT_IPS = ["192.168.1.10", "192.168.1.20", "192.168.1.30"]
    GATEWAY    = "10.0.0.1"

    print("Generating test traffic scenarios...")

    # ── HTTPS sessions with various SNIs ──────────────────────────────────────
    https_sites = [
        ("www.youtube.com",   "192.168.1.10"),
        ("www.facebook.com",  "192.168.1.10"),
        ("www.instagram.com", "192.168.1.20"),
        ("www.tiktok.com",    "192.168.1.20"),
        ("www.netflix.com",   "192.168.1.10"),
        ("github.com",        "192.168.1.30"),
        ("www.google.com",    "192.168.1.10"),
        ("discord.com",       "192.168.1.30"),
        ("www.reddit.com",    "192.168.1.20"),
        ("open.spotify.com",  "192.168.1.10"),
        ("www.youtube.com",   "192.168.1.20"),  # Multiple sessions to same site
        ("www.facebook.com",  "192.168.1.30"),
    ]
    for sni, client in https_sites:
        sport = random.randint(40000, 60000)
        ts = scenario_https_with_sni(packets, ts, client, GATEWAY, sni, sport)
        ts += random.uniform(0.5, 2.0)
        print(f"  ✓ HTTPS: {sni} from {client}")

    # ── HTTP sessions ─────────────────────────────────────────────────────────
    http_sites = [
        ("example.com",    "192.168.1.10", "93.184.216.34"),
        ("httpbin.org",    "192.168.1.20", "54.235.20.216"),
        ("neverssl.com",   "192.168.1.30", "1.2.3.100"),
    ]
    for host, client, server_ip in http_sites:
        sport = random.randint(40000, 60000)
        ts = scenario_http_request(packets, ts, client, server_ip, host, sport)
        ts += 1.0
        print(f"  ✓ HTTP: {host}")

    # ── DNS queries ───────────────────────────────────────────────────────────
    ts = scenario_dns_queries(packets, ts, "192.168.1.10")
    print("  ✓ DNS queries (6 domains)")

    # ── Port scan (anomalous — for ML testing) ────────────────────────────────
    ts = scenario_port_scan(packets, ts, "192.168.1.99")
    print("  ✓ Port scan from 192.168.1.99 (50 ports)")

    # ── High-volume burst (potential exfiltration) ────────────────────────────
    ts = scenario_high_volume(packets, ts, "192.168.1.50", "10.0.0.200")
    print("  ✓ High-volume burst from 192.168.1.50 (exfiltration sim)")

    # ── Write PCAP ────────────────────────────────────────────────────────────
    # Sort by timestamp
    packets.sort(key=lambda x: x[0])

    with open(output_path, "wb") as f:
        f.write(pcap_global_header())
        for pkt_ts, pkt_data in packets:
            f.write(pcap_packet_header(pkt_data, pkt_ts))
            f.write(pkt_data)

    size_kb = os.path.getsize(output_path) / 1024
    print(f"\n✅ Written {len(packets)} packets to {output_path} ({size_kb:.1f} KB)")
    return len(packets)


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "test_traffic.pcap"
    generate_test_pcap(out)