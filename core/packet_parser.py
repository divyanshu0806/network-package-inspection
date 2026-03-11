"""
core/packet_parser.py
---------------------
Pure-Python network protocol parser.

Parses raw packet bytes through the network stack:
  Ethernet → IPv4/IPv6 → TCP / UDP → Payload

Packet structure (nested like Russian dolls):
┌──────────────────────────────────────────────────────────┐
│ Ethernet Header  (14 bytes)                              │
│  ┌────────────────────────────────────────────────────┐  │
│  │ IP Header  (20+ bytes)                             │  │
│  │  ┌──────────────────────────────────────────────┐  │  │
│  │  │ TCP/UDP Header  (20 / 8 bytes)               │  │  │
│  │  │  ┌────────────────────────────────────────┐  │  │  │
│  │  │  │ Payload (application data)             │  │  │  │
│  │  │  └────────────────────────────────────────┘  │  │  │
│  │  └──────────────────────────────────────────────┘  │  │
│  └────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────┘
"""

import struct
import socket
from dataclasses import dataclass, field
from typing import Optional
from core.pcap_reader import RawPacket


# ── EtherType Constants ───────────────────────────────────────────────────────
ETHERTYPE_IPv4 = 0x0800
ETHERTYPE_ARP  = 0x0806
ETHERTYPE_IPv6 = 0x86DD
ETHERTYPE_VLAN = 0x8100   # 802.1Q VLAN tag

# ── IP Protocol Numbers ───────────────────────────────────────────────────────
PROTO_ICMP = 1
PROTO_TCP  = 6
PROTO_UDP  = 17

# ── Well-Known Port Labels ────────────────────────────────────────────────────
PORT_LABELS = {
    20:   "FTP-data",
    21:   "FTP",
    22:   "SSH",
    23:   "Telnet",
    25:   "SMTP",
    53:   "DNS",
    67:   "DHCP-server",
    68:   "DHCP-client",
    80:   "HTTP",
    110:  "POP3",
    143:  "IMAP",
    443:  "HTTPS",
    465:  "SMTPS",
    587:  "SMTP-TLS",
    853:  "DNS-TLS",
    993:  "IMAPS",
    995:  "POP3S",
    3306: "MySQL",
    5222: "XMPP",
    8080: "HTTP-alt",
    8443: "HTTPS-alt",
}


# ── Data Structures ──────────────────────────────────────────────────────────

@dataclass
class EthernetHeader:
    dst_mac:   str          # "aa:bb:cc:dd:ee:ff"
    src_mac:   str
    ethertype: int          # 0x0800 = IPv4, 0x86DD = IPv6
    vlan_id:   Optional[int] = None   # Set if 802.1Q VLAN tag present


@dataclass
class IPv4Header:
    version:    int         # Should be 4
    ihl:        int         # Header length in 32-bit words (usually 5 = 20 bytes)
    dscp:       int         # DSCP / QoS field
    ecn:        int
    total_len:  int         # Total IP packet length
    ident:      int         # Fragmentation identifier
    flags:      int         # DF, MF bits
    frag_off:   int         # Fragment offset
    ttl:        int         # Time-to-live
    protocol:   int         # 6=TCP, 17=UDP, 1=ICMP
    checksum:   int
    src_ip:     str         # "192.168.1.1"
    dst_ip:     str

    @property
    def header_len(self) -> int:
        return self.ihl * 4

    @property
    def protocol_name(self) -> str:
        return {PROTO_TCP: "TCP", PROTO_UDP: "UDP", PROTO_ICMP: "ICMP"}.get(
            self.protocol, f"PROTO_{self.protocol}"
        )


@dataclass
class IPv6Header:
    version:      int       # Should be 6
    traffic_class: int
    flow_label:   int
    payload_len:  int
    next_header:  int       # Protocol of next header (TCP=6, UDP=17)
    hop_limit:    int
    src_ip:       str       # Full IPv6 address string
    dst_ip:       str

    @property
    def protocol(self) -> int:
        return self.next_header

    @property
    def protocol_name(self) -> str:
        return {PROTO_TCP: "TCP", PROTO_UDP: "UDP"}.get(
            self.protocol, f"PROTO_{self.protocol}"
        )


@dataclass
class TCPHeader:
    src_port:  int
    dst_port:  int
    seq_num:   int
    ack_num:   int
    data_off:  int          # Header length in 32-bit words
    flags:     int          # SYN, ACK, FIN, RST, PSH, URG
    window:    int
    checksum:  int
    urgent:    int

    # Flag bitmasks
    FLAG_FIN = 0x01
    FLAG_SYN = 0x02
    FLAG_RST = 0x04
    FLAG_PSH = 0x08
    FLAG_ACK = 0x10
    FLAG_URG = 0x20

    @property
    def header_len(self) -> int:
        return self.data_off * 4

    @property
    def flag_str(self) -> str:
        names = []
        if self.flags & self.FLAG_SYN: names.append("SYN")
        if self.flags & self.FLAG_ACK: names.append("ACK")
        if self.flags & self.FLAG_FIN: names.append("FIN")
        if self.flags & self.FLAG_RST: names.append("RST")
        if self.flags & self.FLAG_PSH: names.append("PSH")
        if self.flags & self.FLAG_URG: names.append("URG")
        return "|".join(names) if names else "-"

    @property
    def is_syn(self) -> bool:
        return bool(self.flags & self.FLAG_SYN) and not (self.flags & self.FLAG_ACK)

    @property
    def is_fin(self) -> bool:
        return bool(self.flags & self.FLAG_FIN)

    @property
    def src_service(self) -> str:
        return PORT_LABELS.get(self.src_port, str(self.src_port))

    @property
    def dst_service(self) -> str:
        return PORT_LABELS.get(self.dst_port, str(self.dst_port))


@dataclass
class UDPHeader:
    src_port: int
    dst_port: int
    length:   int
    checksum: int

    @property
    def src_service(self) -> str:
        return PORT_LABELS.get(self.src_port, str(self.src_port))

    @property
    def dst_service(self) -> str:
        return PORT_LABELS.get(self.dst_port, str(self.dst_port))


@dataclass
class ParsedPacket:
    """
    Fully decoded packet with all layer headers and payload.
    """
    raw:      RawPacket

    # Layer 2
    ethernet: Optional[EthernetHeader] = None

    # Layer 3
    ipv4:     Optional[IPv4Header]     = None
    ipv6:     Optional[IPv6Header]     = None

    # Layer 4
    tcp:      Optional[TCPHeader]      = None
    udp:      Optional[UDPHeader]      = None

    # Layer 7 payload bytes
    payload:  bytes = b""

    # Parse errors / notes
    errors:   list  = field(default_factory=list)

    # ── Convenience Properties ────────────────────────────────────────────────

    @property
    def src_ip(self) -> str:
        if self.ipv4:  return self.ipv4.src_ip
        if self.ipv6:  return self.ipv6.src_ip
        return ""

    @property
    def dst_ip(self) -> str:
        if self.ipv4:  return self.ipv4.dst_ip
        if self.ipv6:  return self.ipv6.dst_ip
        return ""

    @property
    def src_port(self) -> int:
        if self.tcp: return self.tcp.src_port
        if self.udp: return self.udp.src_port
        return 0

    @property
    def dst_port(self) -> int:
        if self.tcp: return self.tcp.dst_port
        if self.udp: return self.udp.dst_port
        return 0

    @property
    def protocol(self) -> int:
        if self.ipv4: return self.ipv4.protocol
        if self.ipv6: return self.ipv6.protocol
        return 0

    @property
    def protocol_name(self) -> str:
        if self.tcp: return "TCP"
        if self.udp: return "UDP"
        if self.ipv4: return self.ipv4.protocol_name
        if self.ipv6: return self.ipv6.protocol_name
        return "UNKNOWN"

    @property
    def is_tcp(self) -> bool:  return self.tcp is not None
    @property
    def is_udp(self) -> bool:  return self.udp is not None
    @property
    def is_ipv4(self) -> bool: return self.ipv4 is not None
    @property
    def is_ipv6(self) -> bool: return self.ipv6 is not None

    @property
    def five_tuple(self) -> tuple:
        """(src_ip, dst_ip, src_port, dst_port, protocol)"""
        return (self.src_ip, self.dst_ip, self.src_port, self.dst_port, self.protocol)

    def summary(self) -> str:
        proto = self.protocol_name
        return (f"[#{self.raw.index}] {self.src_ip}:{self.src_port} → "
                f"{self.dst_ip}:{self.dst_port}  {proto}  "
                f"{len(self.payload)}B payload")


# ── PacketParser ──────────────────────────────────────────────────────────────

class PacketParser:
    """
    Stateless parser: takes a RawPacket, returns a ParsedPacket.

    Usage:
        parser = PacketParser()
        parsed = parser.parse(raw_pkt)
        print(parsed.src_ip, parsed.dst_ip)
    """

    @staticmethod
    def parse(raw: RawPacket) -> ParsedPacket:
        """
        Parse all protocol layers from raw bytes.
        Returns ParsedPacket (errors list is populated on failures).
        """
        pkt = ParsedPacket(raw=raw)
        data = raw.data

        if len(data) < 14:
            pkt.errors.append("Packet too short for Ethernet header")
            return pkt

        # ── Layer 2: Ethernet ─────────────────────────────────────────────────
        offset = PacketParser._parse_ethernet(data, pkt)
        if offset < 0:
            return pkt

        ethertype = pkt.ethernet.ethertype

        # ── Layer 3: IP ───────────────────────────────────────────────────────
        if ethertype == ETHERTYPE_IPv4:
            offset = PacketParser._parse_ipv4(data, offset, pkt)
            if offset < 0:
                return pkt
            ip_proto = pkt.ipv4.protocol

        elif ethertype == ETHERTYPE_IPv6:
            offset = PacketParser._parse_ipv6(data, offset, pkt)
            if offset < 0:
                return pkt
            ip_proto = pkt.ipv6.next_header

        else:
            # ARP, or other — no IP layer to parse
            pkt.payload = data[offset:]
            return pkt

        # ── Layer 4: TCP / UDP ────────────────────────────────────────────────
        if ip_proto == PROTO_TCP:
            offset = PacketParser._parse_tcp(data, offset, pkt)
        elif ip_proto == PROTO_UDP:
            offset = PacketParser._parse_udp(data, offset, pkt)

        # ── Payload ───────────────────────────────────────────────────────────
        if offset >= 0 and offset <= len(data):
            pkt.payload = data[offset:]

        return pkt

    # ── Private Layer Parsers ─────────────────────────────────────────────────

    @staticmethod
    def _parse_ethernet(data: bytes, pkt: ParsedPacket) -> int:
        """Parse Ethernet II header. Returns offset after header."""
        try:
            dst_mac = PacketParser._fmt_mac(data[0:6])
            src_mac = PacketParser._fmt_mac(data[6:12])
            ethertype = struct.unpack("!H", data[12:14])[0]
            offset = 14
            vlan_id = None

            # Handle 802.1Q VLAN tag (4 extra bytes)
            if ethertype == ETHERTYPE_VLAN:
                if len(data) < 18:
                    pkt.errors.append("VLAN tag truncated")
                    return -1
                tci = struct.unpack("!H", data[14:16])[0]
                vlan_id = tci & 0x0FFF
                ethertype = struct.unpack("!H", data[16:18])[0]
                offset = 18

            pkt.ethernet = EthernetHeader(
                dst_mac=dst_mac,
                src_mac=src_mac,
                ethertype=ethertype,
                vlan_id=vlan_id,
            )
            return offset

        except struct.error as e:
            pkt.errors.append(f"Ethernet parse error: {e}")
            return -1

    @staticmethod
    def _parse_ipv4(data: bytes, offset: int, pkt: ParsedPacket) -> int:
        """Parse IPv4 header. Returns offset after header (start of L4)."""
        try:
            if len(data) - offset < 20:
                pkt.errors.append("IPv4 header truncated")
                return -1

            b0 = data[offset]
            version = (b0 >> 4) & 0xF
            ihl     = b0 & 0xF          # header length in 32-bit words
            dscp_ecn = data[offset + 1]
            dscp = (dscp_ecn >> 2)
            ecn  = dscp_ecn & 0x3
            total_len, ident, flags_frag, ttl, proto, checksum = struct.unpack(
                "!HHHBBH", data[offset+2 : offset+12]
            )
            flags    = (flags_frag >> 13) & 0x7
            frag_off = flags_frag & 0x1FFF
            src_ip = socket.inet_ntoa(data[offset+12 : offset+16])
            dst_ip = socket.inet_ntoa(data[offset+16 : offset+20])

            pkt.ipv4 = IPv4Header(
                version=version, ihl=ihl, dscp=dscp, ecn=ecn,
                total_len=total_len, ident=ident, flags=flags,
                frag_off=frag_off, ttl=ttl, protocol=proto,
                checksum=checksum, src_ip=src_ip, dst_ip=dst_ip,
            )
            return offset + ihl * 4     # skip any IP options

        except (struct.error, OSError) as e:
            pkt.errors.append(f"IPv4 parse error: {e}")
            return -1

    @staticmethod
    def _parse_ipv6(data: bytes, offset: int, pkt: ParsedPacket) -> int:
        """Parse IPv6 header (fixed 40 bytes)."""
        try:
            if len(data) - offset < 40:
                pkt.errors.append("IPv6 header truncated")
                return -1

            first_word = struct.unpack("!I", data[offset:offset+4])[0]
            version       = (first_word >> 28) & 0xF
            traffic_class = (first_word >> 20) & 0xFF
            flow_label    = first_word & 0xFFFFF
            payload_len, next_hdr, hop_limit = struct.unpack(
                "!HBB", data[offset+4:offset+8]
            )
            src_ip = socket.inet_ntop(socket.AF_INET6, data[offset+8  : offset+24])
            dst_ip = socket.inet_ntop(socket.AF_INET6, data[offset+24 : offset+40])

            pkt.ipv6 = IPv6Header(
                version=version, traffic_class=traffic_class,
                flow_label=flow_label, payload_len=payload_len,
                next_header=next_hdr, hop_limit=hop_limit,
                src_ip=src_ip, dst_ip=dst_ip,
            )
            return offset + 40

        except (struct.error, OSError) as e:
            pkt.errors.append(f"IPv6 parse error: {e}")
            return -1

    @staticmethod
    def _parse_tcp(data: bytes, offset: int, pkt: ParsedPacket) -> int:
        """Parse TCP header. Returns offset after header (start of payload)."""
        try:
            if len(data) - offset < 20:
                pkt.errors.append("TCP header truncated")
                return -1

            src_port, dst_port, seq, ack = struct.unpack(
                "!HHII", data[offset:offset+12]
            )
            data_off_flags = struct.unpack("!H", data[offset+12:offset+14])[0]
            data_off = (data_off_flags >> 12) & 0xF
            flags    = data_off_flags & 0x1FF   # 9 flag bits
            window, checksum, urgent = struct.unpack(
                "!HHH", data[offset+14:offset+20]
            )

            pkt.tcp = TCPHeader(
                src_port=src_port, dst_port=dst_port,
                seq_num=seq, ack_num=ack, data_off=data_off,
                flags=flags, window=window,
                checksum=checksum, urgent=urgent,
            )
            return offset + data_off * 4    # skip TCP options

        except struct.error as e:
            pkt.errors.append(f"TCP parse error: {e}")
            return -1

    @staticmethod
    def _parse_udp(data: bytes, offset: int, pkt: ParsedPacket) -> int:
        """Parse UDP header (8 bytes fixed). Returns offset after header."""
        try:
            if len(data) - offset < 8:
                pkt.errors.append("UDP header truncated")
                return -1

            src_port, dst_port, length, checksum = struct.unpack(
                "!HHHH", data[offset:offset+8]
            )
            pkt.udp = UDPHeader(
                src_port=src_port, dst_port=dst_port,
                length=length, checksum=checksum,
            )
            return offset + 8

        except struct.error as e:
            pkt.errors.append(f"UDP parse error: {e}")
            return -1

    # ── Utilities ─────────────────────────────────────────────────────────────

    @staticmethod
    def _fmt_mac(b: bytes) -> str:
        return ":".join(f"{x:02x}" for x in b)