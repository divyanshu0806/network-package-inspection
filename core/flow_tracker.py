"""
core/flow_tracker.py
--------------------
Stateful network flow tracking using 5-tuple session keys.

A "flow" is a bidirectional network conversation identified by:
  (src_ip, dst_ip, src_port, dst_port, protocol)

Why flow tracking matters:
  A single TCP connection spans many packets:
    SYN → SYN-ACK → ACK → [data packets] → FIN → FIN-ACK
  We need to track ALL packets as one session to:
    - Build per-flow statistics for ML feature extraction
    - Associate a domain name (SNI) with ALL packets in the session
    - Block/allow at the flow level, not just packet level

Bidirectional flows:
  A → B and B → A are the SAME conversation.
  We normalize the key so the flow is always the same regardless of direction:
    key = tuple(sorted([(src_ip, src_port), (dst_ip, dst_port)]) + [protocol])
"""

import time
from dataclasses import dataclass, field
from typing import Optional, Dict, List
from enum import Enum

from core.packet_parser import ParsedPacket, PROTO_TCP, PROTO_UDP


# ── Flow State ────────────────────────────────────────────────────────────────

class FlowState(Enum):
    NEW       = "NEW"        # Just created, first SYN seen
    ACTIVE    = "ACTIVE"     # Data flowing
    CLOSING   = "CLOSING"    # FIN seen
    CLOSED    = "CLOSED"     # FIN-ACK or RST seen
    EXPIRED   = "EXPIRED"    # Timed out (no packets for a while)
    BLOCKED   = "BLOCKED"    # Blocked by rule engine


# ── App Classification ────────────────────────────────────────────────────────

class AppType(Enum):
    UNKNOWN    = "Unknown"
    HTTP       = "HTTP"
    HTTPS      = "HTTPS"
    DNS        = "DNS"
    YOUTUBE    = "YouTube"
    FACEBOOK   = "Facebook"
    INSTAGRAM  = "Instagram"
    TWITTER    = "Twitter / X"
    TIKTOK     = "TikTok"
    GOOGLE     = "Google"
    NETFLIX    = "Netflix"
    AMAZON     = "Amazon"
    MICROSOFT  = "Microsoft"
    APPLE      = "Apple"
    CLOUDFLARE = "Cloudflare"
    GITHUB     = "GitHub"
    WHATSAPP   = "WhatsApp"
    TELEGRAM   = "Telegram"
    ZOOM       = "Zoom"
    DISCORD    = "Discord"
    REDDIT     = "Reddit"
    TWITCH     = "Twitch"
    SPOTIFY    = "Spotify"


# Domain → AppType lookup table
_SNI_TO_APP: List[tuple] = [
    # Keywords are matched as substrings of the SNI
    ("youtube",     AppType.YOUTUBE),
    ("googlevideo", AppType.YOUTUBE),
    ("youtu.be",    AppType.YOUTUBE),
    ("facebook",    AppType.FACEBOOK),
    ("fbcdn",       AppType.FACEBOOK),
    ("instagram",   AppType.INSTAGRAM),
    ("cdninstagram",AppType.INSTAGRAM),
    ("twitter",     AppType.TWITTER),
    ("twimg",       AppType.TWITTER),
    ("t.co",        AppType.TWITTER),
    ("tiktok",      AppType.TIKTOK),
    ("tiktokcdn",   AppType.TIKTOK),
    ("netflix",     AppType.NETFLIX),
    ("nflxvideo",   AppType.NETFLIX),
    ("amazon",      AppType.AMAZON),
    ("amazonaws",   AppType.AMAZON),
    ("microsoft",   AppType.MICROSOFT),
    ("msftconnect", AppType.MICROSOFT),
    ("windows.com", AppType.MICROSOFT),
    ("office",      AppType.MICROSOFT),
    ("apple",       AppType.APPLE),
    ("icloud",      AppType.APPLE),
    ("akamai",      AppType.CLOUDFLARE),
    ("cloudflare",  AppType.CLOUDFLARE),
    ("github",      AppType.GITHUB),
    ("whatsapp",    AppType.WHATSAPP),
    ("wa.me",       AppType.WHATSAPP),
    ("telegram",    AppType.TELEGRAM),
    ("zoom.us",     AppType.ZOOM),
    ("discord",     AppType.DISCORD),
    ("reddit",      AppType.REDDIT),
    ("redd.it",     AppType.REDDIT),
    ("twitch",      AppType.TWITCH),
    ("spotify",     AppType.SPOTIFY),
    ("google",      AppType.GOOGLE),
    ("gstatic",     AppType.GOOGLE),
    ("googleapis",  AppType.GOOGLE),
]


def classify_domain(domain: str) -> AppType:
    """Map a domain/SNI string to an AppType."""
    if not domain:
        return AppType.UNKNOWN
    low = domain.lower()
    for keyword, app in _SNI_TO_APP:
        if keyword in low:
            return app
    return AppType.UNKNOWN


def classify_by_port(dst_port: int, src_port: int) -> AppType:
    """Fallback classification using port numbers."""
    if dst_port == 53 or src_port == 53:
        return AppType.DNS
    if dst_port in (80, 8080, 8000):
        return AppType.HTTP
    if dst_port in (443, 8443, 853):
        return AppType.HTTPS
    return AppType.UNKNOWN


# ── Flow Record ───────────────────────────────────────────────────────────────

@dataclass
class Flow:
    """
    Represents a single bidirectional network flow (session).
    All packets belonging to the same connection update this record.
    """
    # Identity
    flow_key:    tuple           # Normalized 5-tuple key
    src_ip:      str
    dst_ip:      str
    src_port:    int
    dst_port:    int
    protocol:    int

    # Timing
    first_seen:  float = field(default_factory=time.time)
    last_seen:   float = field(default_factory=time.time)

    # Packet statistics
    pkt_count:   int   = 0       # Total packets (both directions)
    byte_count:  int   = 0       # Total bytes (both directions)
    fwd_pkts:    int   = 0       # Forward direction packets (src→dst)
    bwd_pkts:    int   = 0       # Backward direction packets
    fwd_bytes:   int   = 0
    bwd_bytes:   int   = 0

    # Packet size stats (for ML features)
    min_pkt_len: int   = 65535
    max_pkt_len: int   = 0
    pkt_lengths: list  = field(default_factory=list)  # last 100 sizes

    # Inter-arrival time stats
    iat_list:    list  = field(default_factory=list)   # inter-arrival times
    _last_pkt_time: float = 0.0

    # TCP-specific
    tcp_flags_seen:  int = 0     # OR of all flags seen
    syn_count:   int = 0
    fin_count:   int = 0
    rst_count:   int = 0

    # DPI results
    sni:         Optional[str]  = None
    http_host:   Optional[str]  = None
    dns_query:   Optional[str]  = None
    tls_version: Optional[str]  = None
    app_type:    AppType        = AppType.UNKNOWN

    # Flow control
    state:       FlowState      = FlowState.NEW
    blocked:     bool           = False
    block_reason: str           = ""
    anomaly_score: float        = 0.0   # Set by ML detector (Phase 3)
    threat_label:  str          = ""    # Set by ML classifier

    # ── Update Methods ─────────────────────────────────────────────────────────

    def update(self, pkt: ParsedPacket) -> None:
        """Update flow stats with a new packet."""
        now = pkt.raw.timestamp
        pkt_len = len(pkt.raw.data)

        # Timing
        if self._last_pkt_time > 0:
            iat = now - self._last_pkt_time
            if len(self.iat_list) < 200:
                self.iat_list.append(iat)
        self._last_pkt_time = now
        self.last_seen = now

        # Counters
        self.pkt_count  += 1
        self.byte_count += pkt_len

        # Direction: is this src→dst or dst→src?
        is_forward = (pkt.src_ip == self.src_ip and
                      pkt.src_port == self.src_port)
        if is_forward:
            self.fwd_pkts  += 1
            self.fwd_bytes += pkt_len
        else:
            self.bwd_pkts  += 1
            self.bwd_bytes += pkt_len

        # Packet size tracking (keep last 100)
        self.min_pkt_len = min(self.min_pkt_len, pkt_len)
        self.max_pkt_len = max(self.max_pkt_len, pkt_len)
        if len(self.pkt_lengths) < 100:
            self.pkt_lengths.append(pkt_len)

        # TCP flag tracking
        if pkt.tcp:
            self.tcp_flags_seen |= pkt.tcp.flags
            if pkt.tcp.is_syn:
                self.syn_count += 1
            if pkt.tcp.is_fin:
                self.fin_count += 1
                self.state = FlowState.CLOSING
            if pkt.tcp.flags & 0x04:  # RST
                self.rst_count += 1
                self.state = FlowState.CLOSED

        # Update state machine
        if self.state == FlowState.NEW and self.pkt_count > 1:
            self.state = FlowState.ACTIVE

    def set_dpi_result(self, domain: str, source: str,
                       tls_version: Optional[str] = None) -> None:
        """Apply DPI inspection result to this flow."""
        if source == "tls":
            self.sni = domain
        elif source == "http":
            self.http_host = domain
        elif source == "dns":
            self.dns_query = domain

        if tls_version:
            self.tls_version = tls_version

        # Classify app type from domain
        if self.app_type == AppType.UNKNOWN:
            self.app_type = classify_domain(domain)

    # ── Computed Properties ────────────────────────────────────────────────────

    @property
    def domain(self) -> Optional[str]:
        return self.sni or self.http_host or self.dns_query

    @property
    def duration(self) -> float:
        return self.last_seen - self.first_seen

    @property
    def avg_pkt_len(self) -> float:
        return self.byte_count / self.pkt_count if self.pkt_count else 0

    @property
    def avg_iat(self) -> float:
        return sum(self.iat_list) / len(self.iat_list) if self.iat_list else 0

    @property
    def protocol_name(self) -> str:
        return {6: "TCP", 17: "UDP", 1: "ICMP"}.get(self.protocol, str(self.protocol))

    def summary(self) -> str:
        domain_str = f"  [{self.domain}]" if self.domain else ""
        blocked_str = "  ⛔ BLOCKED" if self.blocked else ""
        return (f"{self.src_ip}:{self.src_port} → {self.dst_ip}:{self.dst_port}"
                f"  {self.protocol_name}  {self.pkt_count}pkts  {self.byte_count}B"
                f"  {self.app_type.value}{domain_str}{blocked_str}")

    def to_dict(self) -> dict:
        """Serialize to dict for export / ML feature extraction."""
        return {
            "src_ip":       self.src_ip,
            "dst_ip":       self.dst_ip,
            "src_port":     self.src_port,
            "dst_port":     self.dst_port,
            "protocol":     self.protocol,
            "protocol_name": self.protocol_name,
            "first_seen":   self.first_seen,
            "last_seen":    self.last_seen,
            "duration":     self.duration,
            "pkt_count":    self.pkt_count,
            "byte_count":   self.byte_count,
            "fwd_pkts":     self.fwd_pkts,
            "bwd_pkts":     self.bwd_pkts,
            "fwd_bytes":    self.fwd_bytes,
            "bwd_bytes":    self.bwd_bytes,
            "avg_pkt_len":  self.avg_pkt_len,
            "min_pkt_len":  self.min_pkt_len,
            "max_pkt_len":  self.max_pkt_len,
            "avg_iat":      self.avg_iat,
            "tcp_flags":    self.tcp_flags_seen,
            "syn_count":    self.syn_count,
            "fin_count":    self.fin_count,
            "rst_count":    self.rst_count,
            "sni":          self.sni,
            "http_host":    self.http_host,
            "dns_query":    self.dns_query,
            "domain":       self.domain,
            "app_type":     self.app_type.value,
            "tls_version":  self.tls_version,
            "state":        self.state.value,
            "blocked":      self.blocked,
            "anomaly_score": self.anomaly_score,
            "threat_label": self.threat_label,
        }


# ── FlowTracker ───────────────────────────────────────────────────────────────

class FlowTracker:
    """
    Maintains a table of all active network flows.

    Normalizes 5-tuples so A→B and B→A map to the same flow.
    Provides flow lookup, creation, aging/expiry, and export.

    Usage:
        tracker = FlowTracker()
        for pkt in parsed_packets:
            flow = tracker.get_or_create(pkt)
            flow.update(pkt)
    """

    # Flow expiry timeouts (seconds)
    TCP_TIMEOUT  = 300    # 5 minutes
    UDP_TIMEOUT  = 30     # 30 seconds
    OTHER_TIMEOUT = 60

    def __init__(self):
        self._flows: Dict[tuple, Flow] = {}
        self._total_created = 0

    # ── Public API ────────────────────────────────────────────────────────────

    def get_or_create(self, pkt: ParsedPacket) -> Flow:
        """
        Return the existing flow for this packet, or create a new one.
        The key is NORMALIZED — A↔B and B↔A always map to the same flow.
        """
        key = self._make_key(pkt)
        if key not in self._flows:
            self._flows[key] = self._create_flow(key, pkt)
            self._total_created += 1
        return self._flows[key]

    def get(self, pkt: ParsedPacket) -> Optional[Flow]:
        """Return flow if it exists, else None (no creation)."""
        return self._flows.get(self._make_key(pkt))

    def expire_old_flows(self, current_time: Optional[float] = None) -> int:
        """
        Remove flows that have been idle longer than their timeout.
        Returns the number of flows removed.
        """
        now = current_time or time.time()
        to_remove = []
        for key, flow in self._flows.items():
            timeout = self._timeout_for(flow)
            if (now - flow.last_seen) > timeout:
                flow.state = FlowState.EXPIRED
                to_remove.append(key)
        for key in to_remove:
            del self._flows[key]
        return len(to_remove)

    def all_flows(self) -> List[Flow]:
        return list(self._flows.values())

    def active_flows(self) -> List[Flow]:
        return [f for f in self._flows.values()
                if f.state not in (FlowState.CLOSED, FlowState.EXPIRED)]

    def blocked_flows(self) -> List[Flow]:
        return [f for f in self._flows.values() if f.blocked]

    @property
    def flow_count(self) -> int:
        return len(self._flows)

    @property
    def total_created(self) -> int:
        return self._total_created

    def export_csv(self, filepath: str) -> int:
        """Export all flows to CSV. Returns number of rows written."""
        flows = self.all_flows()
        if not flows:
            return 0
        import csv
        fieldnames = list(flows[0].to_dict().keys())
        with open(filepath, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for flow in flows:
                writer.writerow(flow.to_dict())
        return len(flows)

    def stats_summary(self) -> dict:
        """Return aggregate statistics across all flows."""
        flows = self.all_flows()
        if not flows:
            return {}

        total_pkts  = sum(f.pkt_count  for f in flows)
        total_bytes = sum(f.byte_count for f in flows)

        # App type breakdown
        from collections import Counter
        app_counts = Counter(f.app_type.value for f in flows)
        domain_counts = Counter(f.domain for f in flows if f.domain)

        # Protocol breakdown
        proto_counts = Counter(f.protocol_name for f in flows)

        return {
            "total_flows":    len(flows),
            "total_packets":  total_pkts,
            "total_bytes":    total_bytes,
            "blocked_flows":  sum(1 for f in flows if f.blocked),
            "app_breakdown":  dict(app_counts.most_common(10)),
            "top_domains":    dict(domain_counts.most_common(10)),
            "protocol_breakdown": dict(proto_counts),
        }

    # ── Private Helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _make_key(pkt: ParsedPacket) -> tuple:
        """
        Create a normalized, bidirectional flow key.

        We sort the (ip, port) pairs so that:
          A:1234 → B:443  and  B:443 → A:1234
        both produce the same key.
        """
        ep1 = (pkt.src_ip, pkt.src_port)
        ep2 = (pkt.dst_ip, pkt.dst_port)
        # Sort endpoints to normalize direction
        if ep1 > ep2:
            ep1, ep2 = ep2, ep1
        return (*ep1, *ep2, pkt.protocol)

    @staticmethod
    def _create_flow(key: tuple, pkt: ParsedPacket) -> Flow:
        """Create a new Flow record from the first packet."""
        # Port-based initial classification
        app = classify_by_port(pkt.dst_port, pkt.src_port)
        return Flow(
            flow_key=key,
            src_ip=pkt.src_ip,
            dst_ip=pkt.dst_ip,
            src_port=pkt.src_port,
            dst_port=pkt.dst_port,
            protocol=pkt.protocol,
            first_seen=pkt.raw.timestamp,
            last_seen=pkt.raw.timestamp,
            _last_pkt_time=pkt.raw.timestamp,
            app_type=app,
        )

    def _timeout_for(self, flow: Flow) -> float:
        if flow.protocol == PROTO_TCP:
            return self.TCP_TIMEOUT
        elif flow.protocol == PROTO_UDP:
            return self.UDP_TIMEOUT
        return self.OTHER_TIMEOUT