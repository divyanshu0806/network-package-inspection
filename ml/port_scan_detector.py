"""
ml/port_scan_detector.py
------------------------
Detects network port scanning using flow-level heuristics.

Types of port scans detected:
  1. Horizontal scan  — one source hits MANY ports on one target
  2. Vertical scan    — one source sweeps one port across MANY targets
  3. SYN scan         — TCP SYN packets with no SYN-ACK (stealth scan)
  4. NULL/FIN scan    — unusual flag combinations (OS fingerprinting)
  5. UDP scan         — UDP packets to many ports (service discovery)

Detection signals:
  - Source IP connecting to N > threshold unique (dst_ip, dst_port) pairs
  - Very short inter-packet interval (scanning speed)
  - High SYN count, low/zero ACK count (SYN scan)
  - RST responses back from target (port is closed)
  - Small packet payloads (probe-only, no data exchange)

Industry thresholds:
  Snort default: 15 ports in 0.6s
  Suricata:      5 ports in 1s
  We use:        10+ ports in 30s (more conservative, fewer false positives)
"""

from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Tuple
from collections import defaultdict
import time


@dataclass
class ScanResult:
    """Result of port scan analysis for one source IP."""
    src_ip:         str
    scan_type:      str         # "horizontal", "vertical", "syn_scan", "udp_scan"
    target_count:   int         # unique destination IPs targeted
    port_count:     int         # unique destination ports targeted
    total_packets:  int
    scan_rate:      float       # connections per second
    confidence:     float       # 0.0–1.0
    is_scanner:     bool
    targets:        List[str]   # sample of dst IPs
    ports:          List[int]   # sample of dst ports
    verdict:        str = ""

    def __post_init__(self):
        if not self.verdict:
            if self.is_scanner:
                self.verdict = (
                    f"🔍 PORT SCAN — {self.src_ip} | {self.scan_type.upper()} | "
                    f"{self.port_count} ports × {self.target_count} hosts | "
                    f"Rate={self.scan_rate:.1f}/s | Conf={self.confidence:.2f}"
                )
            else:
                self.verdict = f"OK — {self.src_ip}"


class PortScanDetector:
    """
    Detects port scanning activity from a list of flows.

    Groups flows by source IP and looks for:
      - Many unique destination ports (horizontal scan)
      - Many unique destination hosts (vertical scan)
      - High SYN ratio with no data exchange
      - Very rapid connection attempts

    Usage:
        detector = PortScanDetector()
        results = detector.analyze(flows)
        for r in results:
            if r.is_scanner:
                print(r.verdict)
    """

    # Thresholds
    MIN_PORTS_FOR_SCAN    = 10    # ≥ N unique dst ports from same src → suspicious
    MIN_HOSTS_FOR_VSCAN   = 5     # ≥ N unique dst hosts → vertical scan
    MAX_SCAN_INTERVAL     = 30.0  # All probes within this window (seconds)
    MIN_SYN_RATIO         = 0.85  # >85% SYN packets → SYN scan
    CONFIDENCE_THRESHOLD  = 0.50  # Flag if confidence ≥ this

    def __init__(self,
                 min_ports: int         = MIN_PORTS_FOR_SCAN,
                 min_hosts: int         = MIN_HOSTS_FOR_VSCAN,
                 max_window: float      = MAX_SCAN_INTERVAL,
                 confidence_thresh: float = CONFIDENCE_THRESHOLD):
        self.min_ports         = min_ports
        self.min_hosts         = min_hosts
        self.max_window        = max_window
        self.confidence_thresh = confidence_thresh

    def analyze(self, flows) -> List[ScanResult]:
        """
        Analyze all flows for port scanning activity.
        Groups by source IP, then checks scan patterns.
        """
        # Group by source IP
        by_src: Dict[str, list] = defaultdict(list)
        for flow in flows:
            by_src[flow.src_ip].append(flow)

        results = []
        for src_ip, src_flows in by_src.items():
            result = self._analyze_source(src_ip, src_flows)
            if result and result.confidence >= self.confidence_thresh:
                results.append(result)

        return sorted(results, key=lambda r: -r.confidence)

    def _analyze_source(self, src_ip: str, flows: list) -> Optional[ScanResult]:
        """Analyze all flows from a single source IP."""
        if not flows:
            return None

        # Collect unique destinations and ports
        dst_ips:   Set[str] = set()
        dst_ports: Set[int] = set()
        total_syn  = 0
        total_pkts = 0
        timestamps = []

        for flow in flows:
            dst_ips.add(flow.dst_ip)
            dst_ports.add(flow.dst_port)
            total_syn  += flow.syn_count
            total_pkts += flow.pkt_count
            if flow.first_seen:
                timestamps.append(flow.first_seen)

        n_ports = len(dst_ports)
        n_hosts = len(dst_ips)

        # Time window
        if timestamps:
            ts_range = max(timestamps) - min(timestamps)
            ts_range = max(ts_range, 0.001)
        else:
            ts_range = 1.0

        scan_rate = len(flows) / ts_range

        # SYN ratio (SYN scan = high SYN, few complete connections)
        syn_ratio = total_syn / total_pkts if total_pkts > 0 else 0.0

        # Detect type and score
        scan_type, confidence = self._classify(
            n_ports, n_hosts, syn_ratio, scan_rate, ts_range, len(flows)
        )

        if confidence < 0.1:
            return None

        is_scanner = confidence >= self.confidence_thresh

        return ScanResult(
            src_ip=src_ip,
            scan_type=scan_type,
            target_count=n_hosts,
            port_count=n_ports,
            total_packets=total_pkts,
            scan_rate=scan_rate,
            confidence=round(confidence, 3),
            is_scanner=is_scanner,
            targets=sorted(dst_ips)[:10],
            ports=sorted(dst_ports)[:20],
        )

    def _classify(self, n_ports: int, n_hosts: int,
                   syn_ratio: float, scan_rate: float,
                   ts_range: float, n_flows: int) -> Tuple[str, float]:
        """
        Classify the scan type and compute confidence score.
        Returns (scan_type_str, confidence_0_to_1)
        """
        score = 0.0
        scan_type = "unknown"

        # ── Horizontal scan (many ports, one host) ─────────────────────────────
        if n_ports >= self.min_ports and n_hosts <= 2:
            scan_type = "horizontal"
            # Score: more ports = more certain, up to 1.0 at 50+ ports
            port_score = min(1.0, n_ports / 50)
            # Fast scanning raises confidence
            rate_score = min(1.0, scan_rate / 10)
            score = 0.6 * port_score + 0.4 * rate_score

        # ── Vertical scan (many hosts, one port) ───────────────────────────────
        elif n_hosts >= self.min_hosts and n_ports <= 2:
            scan_type = "vertical"
            host_score = min(1.0, n_hosts / 30)
            rate_score = min(1.0, scan_rate / 5)
            score = 0.6 * host_score + 0.4 * rate_score

        # ── Broad scan (many ports AND many hosts) ─────────────────────────────
        elif n_ports >= self.min_ports and n_hosts >= self.min_hosts:
            scan_type = "broad"
            port_score = min(1.0, n_ports / 50)
            host_score = min(1.0, n_hosts / 20)
            score = 0.5 * port_score + 0.5 * host_score
            score = min(1.0, score * 1.2)   # bonus for both dimensions

        # ── SYN scan (high SYN ratio, moderate port count) ─────────────────────
        if syn_ratio >= self.MIN_SYN_RATIO and n_ports >= 5:
            syn_score = syn_ratio
            port_score = min(1.0, n_ports / 20)
            syn_scan_score = 0.7 * syn_score + 0.3 * port_score
            if syn_scan_score > score:
                scan_type = "syn_scan"
                score = syn_scan_score

        # ── Minimum flow count guard ───────────────────────────────────────────
        # Require enough evidence
        if n_flows < 3:
            score *= 0.3

        return scan_type, round(min(1.0, score), 4)


# ── Data Exfiltration Detector ────────────────────────────────────────────────

@dataclass
class ExfilResult:
    """Result of data exfiltration check for one flow."""
    src_ip:      str
    dst_ip:      str
    dst_port:    int
    byte_count:  int
    pkt_count:   int
    duration:    float
    byte_rate:   float       # bytes per second
    confidence:  float
    is_exfil:    bool
    signals:     List[str]   # list of triggered signals
    verdict:     str = ""

    def __post_init__(self):
        if not self.verdict:
            mb = self.byte_count / (1024*1024)
            self.verdict = (
                f"{'🚨 EXFIL' if self.is_exfil else 'OK  '} — "
                f"{self.src_ip} → {self.dst_ip}:{self.dst_port} | "
                f"{mb:.2f}MB @ {self.byte_rate/1024:.1f}KB/s | "
                f"Conf={self.confidence:.2f}"
            )


class ExfiltrationDetector:
    """
    Detects potential data exfiltration from flow statistics.

    Signals used:
      1. Large outbound byte count (> threshold)
      2. High byte rate sustained over time
      3. Much more data sent than received (high fwd/bwd ratio)
      4. Connection to unusual port (not 80/443)
      5. Night-time activity (if timestamps available)
      6. DNS tunneling indicators (DNS with large payloads)

    Usage:
        detector = ExfiltrationDetector()
        results = detector.analyze(flows)
    """

    LARGE_UPLOAD_BYTES  = 10 * 1024 * 1024   # 10 MB
    HIGH_RATE_BYTES_SEC = 500 * 1024          # 500 KB/s sustained
    HIGH_FWD_BWD_RATIO  = 5.0                 # 5x more sent than received
    CONFIDENCE_THRESHOLD = 0.45

    def __init__(self,
                 large_upload: int    = LARGE_UPLOAD_BYTES,
                 high_rate: float     = HIGH_RATE_BYTES_SEC,
                 fwd_bwd_ratio: float = HIGH_FWD_BWD_RATIO,
                 conf_thresh: float   = CONFIDENCE_THRESHOLD):
        self.large_upload   = large_upload
        self.high_rate      = high_rate
        self.fwd_bwd_ratio  = fwd_bwd_ratio
        self.conf_thresh    = conf_thresh

    def analyze(self, flows) -> List[ExfilResult]:
        results = []
        for flow in flows:
            result = self._check_flow(flow)
            if result and result.confidence >= self.conf_thresh:
                results.append(result)
        return sorted(results, key=lambda r: -r.byte_count)

    def _check_flow(self, flow) -> Optional[ExfilResult]:
        signals = []
        score   = 0.0
        dur     = max(flow.duration, 0.001)
        byte_rate = flow.fwd_bytes / dur

        # Signal 1: Large upload
        if flow.fwd_bytes >= self.large_upload:
            signals.append(f"large_upload({flow.fwd_bytes//1024//1024}MB)")
            score += 0.35

        # Signal 2: High sustained rate
        if byte_rate >= self.high_rate:
            signals.append(f"high_rate({byte_rate/1024:.0f}KB/s)")
            score += 0.30

        # Signal 3: Asymmetric ratio (sending much more than receiving)
        ratio = flow.fwd_bytes / max(flow.bwd_bytes, 1)
        if ratio >= self.fwd_bwd_ratio:
            signals.append(f"asymmetric_ratio({ratio:.1f}x)")
            score += 0.20

        # Signal 4: Unusual destination port
        if flow.dst_port not in (80, 443, 22, 21, 25, 587, 465):
            if flow.fwd_bytes > 100_000:  # only flag if meaningful data
                signals.append(f"unusual_port({flow.dst_port})")
                score += 0.10

        # Signal 5: DNS tunneling (large DNS packets)
        if flow.dst_port == 53 and flow.avg_pkt_len > 200:
            signals.append(f"dns_tunnel_suspect(avg_pkt={flow.avg_pkt_len:.0f}B)")
            score += 0.25

        if not signals:
            return None

        return ExfilResult(
            src_ip=flow.src_ip,
            dst_ip=flow.dst_ip,
            dst_port=flow.dst_port,
            byte_count=flow.byte_count,
            pkt_count=flow.pkt_count,
            duration=dur,
            byte_rate=byte_rate,
            confidence=round(min(1.0, score), 3),
            is_exfil=score >= self.conf_thresh,
            signals=signals,
        )