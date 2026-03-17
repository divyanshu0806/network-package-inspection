"""
core/feature_extractor.py
--------------------------
Transforms raw Flow objects into numeric feature vectors for ML.

Why feature engineering matters:
  ML models can't directly consume a Flow object — they need a fixed-size
  numeric vector. We compute ~45 statistical features per flow that capture
  patterns ML can learn from:

  ┌─────────────────────────────────────────────────────────────────────┐
  │  FEATURE CATEGORY        │  EXAMPLES                               │
  ├─────────────────────────────────────────────────────────────────────┤
  │  Volume                  │  pkt_count, byte_count, bytes_per_pkt  │
  │  Timing                  │  duration, avg_iat, iat_std             │
  │  Direction ratio         │  fwd/bwd ratio, bytes asymmetry         │
  │  Packet size stats        │  min, max, mean, std, percentiles       │
  │  TCP behavior            │  flag counts, SYN ratio, retransmits    │
  │  Port / protocol signals  │  well-known port?, is_HTTPS?           │
  │  IAT (inter-arrival time) │  mean, std, coefficient of variation   │
  │  Burst detection         │  max burst size, burst frequency        │
  │  JA3 TLS fingerprint      │  16-byte MD5 of TLS params             │
  └─────────────────────────────────────────────────────────────────────┘

These 45 features will feed directly into the Isolation Forest (Phase 3)
for anomaly detection and the Random Forest classifier for threat labeling.

JA3 Fingerprinting:
  JA3 is a TLS client fingerprinting method. It hashes:
    TLS version + Cipher suites + Extensions + Elliptic curves + EC formats
  into a 32-char MD5 string. Different malware families have known JA3 hashes,
  making it a powerful threat intel signal.
"""

import math
import hashlib
import statistics
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

from core.flow_tracker import Flow
from core.sni_extractor import TLSSNIExtractor


# ── Feature Vector ─────────────────────────────────────────────────────────────

# The canonical ordered list of feature names.
# This MUST stay stable — changing order breaks saved ML models.
FEATURE_NAMES = [
    # ── Volume features ────────────────────────────────────────────────────────
    "pkt_count",            # 0   total packets in flow
    "byte_count",           # 1   total bytes
    "bytes_per_pkt",        # 2   avg bytes per packet
    "pkt_rate",             # 3   packets per second
    "byte_rate",            # 4   bytes per second

    # ── Direction features ─────────────────────────────────────────────────────
    "fwd_pkt_count",        # 5   forward direction packets
    "bwd_pkt_count",        # 6   backward direction packets
    "fwd_byte_count",       # 7   forward bytes
    "bwd_byte_count",       # 8   backward bytes
    "fwd_bwd_pkt_ratio",    # 9   fwd/bwd packet ratio (0 if bwd=0 → use 999)
    "fwd_bwd_byte_ratio",   # 10  fwd/bwd byte ratio
    "bwd_fwd_byte_diff",    # 11  |fwd_bytes - bwd_bytes| (upload/download asymmetry)

    # ── Packet size features ───────────────────────────────────────────────────
    "pkt_len_min",          # 12  min packet length
    "pkt_len_max",          # 13  max packet length
    "pkt_len_mean",         # 14  mean packet length
    "pkt_len_std",          # 15  std deviation of packet lengths
    "pkt_len_cv",           # 16  coefficient of variation (std/mean)
    "pkt_len_p25",          # 17  25th percentile
    "pkt_len_p50",          # 18  50th percentile (median)
    "pkt_len_p75",          # 19  75th percentile
    "pkt_len_p90",          # 20  90th percentile

    # ── Timing features ────────────────────────────────────────────────────────
    "duration",             # 21  flow duration in seconds
    "iat_mean",             # 22  mean inter-arrival time (seconds)
    "iat_std",              # 23  std of IAT
    "iat_min",              # 24  min IAT
    "iat_max",              # 25  max IAT
    "iat_cv",               # 26  IAT coefficient of variation

    # ── TCP flag features ──────────────────────────────────────────────────────
    "tcp_syn_count",        # 27  SYN packets seen
    "tcp_fin_count",        # 28  FIN packets seen
    "tcp_rst_count",        # 29  RST packets seen
    "has_syn",              # 30  1 if SYN seen
    "has_rst",              # 31  1 if RST seen (connection reset = suspicious)
    "syn_fin_ratio",        # 32  syn_count / (fin_count+1) — high = potential SYN flood

    # ── Port / service features ────────────────────────────────────────────────
    "dst_port",             # 33  destination port (raw number)
    "src_port",             # 34  source port
    "is_well_known_port",   # 35  1 if dst_port < 1024
    "is_https",             # 36  1 if dst_port == 443
    "is_http",              # 37  1 if dst_port == 80
    "is_dns",               # 38  1 if dst_port == 53
    "is_ephemeral_src",     # 39  1 if src_port >= 49152

    # ── Protocol features ──────────────────────────────────────────────────────
    "is_tcp",               # 40  1 if TCP
    "is_udp",               # 41  1 if UDP

    # ── Application layer features ─────────────────────────────────────────────
    "has_sni",              # 42  1 if SNI was extracted
    "has_http_host",        # 43  1 if HTTP Host extracted
    "has_dns_query",        # 44  1 if DNS query extracted
]

NUM_FEATURES = len(FEATURE_NAMES)   # 45


@dataclass
class FeatureVector:
    """
    Numeric feature vector for a single flow.

    Carries both the raw float array (for ML) and metadata
    (for reporting and investigation).
    """
    flow_id:    str             # "src_ip:sport->dst_ip:dport/proto"
    features:   List[float]     # Exactly NUM_FEATURES floats
    label:      str = ""        # Ground-truth label (if known)
    anomaly_score: float = 0.0  # Set by ML detector
    is_anomaly: bool = False    # Set by ML detector
    ja3:        str = ""        # JA3 fingerprint string (if available)
    ja3_hash:   str = ""        # MD5 of JA3 string

    def __getitem__(self, name_or_idx):
        if isinstance(name_or_idx, str):
            return self.features[FEATURE_NAMES.index(name_or_idx)]
        return self.features[name_or_idx]

    def to_dict(self) -> Dict[str, Any]:
        d = {"flow_id": self.flow_id, "label": self.label,
             "anomaly_score": self.anomaly_score, "is_anomaly": self.is_anomaly,
             "ja3": self.ja3, "ja3_hash": self.ja3_hash}
        for name, val in zip(FEATURE_NAMES, self.features):
            d[name] = val
        return d


# ── JA3 Fingerprinter ─────────────────────────────────────────────────────────

class JA3Fingerprinter:
    """
    Computes JA3 TLS client fingerprints from raw TLS ClientHello bytes.

    JA3 = MD5(TLSVersion,Ciphers,Extensions,EllipticCurves,EllipticCurveFormats)

    Known malicious JA3 hashes can be looked up against threat-intel feeds.
    Example:
      Emotet malware:    e7d705a3286e19ea42f587b6798e1745
      Metasploit Meterp: a0e9f5d64349fb13191bc781f81f42e1
      Cobalt Strike:     72a589da586844d7f0818ce684948eea
    """

    # GREASE values to ignore (RFC 8701 — random extension probing)
    GREASE = {0x0a0a, 0x1a1a, 0x2a2a, 0x3a3a, 0x4a4a, 0x5a5a,
              0x6a6a, 0x7a7a, 0x8a8a, 0x9a9a, 0xaaaa, 0xbaba,
              0xcaca, 0xdada, 0xeaea, 0xfafa}

    @classmethod
    def compute(cls, payload: bytes) -> Optional[tuple]:
        """
        Parse a TLS ClientHello and return (ja3_string, ja3_hash) or None.

        JA3 string format:
          "TLSVersion,Ciphers,Extensions,EllipticCurves,ECPointFormats"
          e.g. "771,4866-4867-4865,0-23-65281-10-11,29-23-24,0"
        """
        import struct
        if not payload or len(payload) < 10:
            return None
        if payload[0] != 0x16 or payload[5] != 0x01:
            return None

        try:
            offset = 9
            # ClientHello version
            if offset + 2 > len(payload): return None
            version = struct.unpack("!H", payload[offset:offset+2])[0]
            offset += 2 + 32    # skip version + random

            # Session ID
            if offset >= len(payload): return None
            sess_len = payload[offset]; offset += 1 + sess_len

            # Cipher suites
            if offset + 2 > len(payload): return None
            cipher_len = struct.unpack("!H", payload[offset:offset+2])[0]
            offset += 2
            ciphers = []
            for i in range(0, cipher_len, 2):
                if offset + 2 > len(payload): break
                c = struct.unpack("!H", payload[offset:offset+2])[0]
                if c not in cls.GREASE:
                    ciphers.append(c)
                offset += 2

            # Compression
            if offset >= len(payload): return None
            comp_len = payload[offset]; offset += 1 + comp_len

            # Extensions
            if offset + 2 > len(payload): return None
            ext_total = struct.unpack("!H", payload[offset:offset+2])[0]
            offset += 2
            ext_end = offset + ext_total

            extensions = []
            curves = []
            ec_formats = []

            while offset + 4 <= ext_end and offset + 4 <= len(payload):
                ext_type = struct.unpack("!H", payload[offset:offset+2])[0]
                ext_len  = struct.unpack("!H", payload[offset+2:offset+4])[0]
                offset += 4
                ext_data = payload[offset:offset+ext_len]

                if ext_type not in cls.GREASE:
                    extensions.append(ext_type)

                # Supported Groups (0x000a)
                if ext_type == 0x000a and len(ext_data) >= 2:
                    gl = struct.unpack("!H", ext_data[0:2])[0]
                    for i in range(2, 2+gl, 2):
                        if i+2 <= len(ext_data):
                            g = struct.unpack("!H", ext_data[i:i+2])[0]
                            if g not in cls.GREASE:
                                curves.append(g)

                # EC Point Formats (0x000b)
                if ext_type == 0x000b and len(ext_data) >= 1:
                    fl = ext_data[0]
                    for i in range(1, 1+fl):
                        if i < len(ext_data):
                            ec_formats.append(ext_data[i])

                offset += ext_len

            ja3_str = (
                f"{version},"
                f"{'-'.join(str(c) for c in ciphers)},"
                f"{'-'.join(str(e) for e in extensions)},"
                f"{'-'.join(str(c) for c in curves)},"
                f"{'-'.join(str(f) for f in ec_formats)}"
            )
            ja3_hash = hashlib.md5(ja3_str.encode()).hexdigest()
            return ja3_str, ja3_hash

        except Exception:
            return None


# ── Feature Extractor ─────────────────────────────────────────────────────────

class FeatureExtractor:
    """
    Converts a Flow object into a FeatureVector with 45 numeric features.

    Usage:
        extractor = FeatureExtractor()
        fv = extractor.extract(flow)
        print(fv.features)  # list of 45 floats
    """

    @staticmethod
    def extract(flow: Flow, tls_payload: bytes = None) -> FeatureVector:
        """
        Compute all features for a flow.

        Args:
            flow:        The Flow object from FlowTracker
            tls_payload: Raw TLS ClientHello bytes (for JA3 computation)
        """
        f = [0.0] * NUM_FEATURES

        # ── Volume features ────────────────────────────────────────────────────
        dur        = max(flow.duration, 1e-9)   # avoid div/0
        f[0]  = float(flow.pkt_count)
        f[1]  = float(flow.byte_count)
        f[2]  = flow.byte_count / flow.pkt_count if flow.pkt_count else 0.0
        f[3]  = flow.pkt_count / dur
        f[4]  = flow.byte_count / dur

        # ── Direction features ─────────────────────────────────────────────────
        f[5]  = float(flow.fwd_pkts)
        f[6]  = float(flow.bwd_pkts)
        f[7]  = float(flow.fwd_bytes)
        f[8]  = float(flow.bwd_bytes)
        f[9]  = (flow.fwd_pkts / flow.bwd_pkts) if flow.bwd_pkts else 999.0
        f[10] = (flow.fwd_bytes / flow.bwd_bytes) if flow.bwd_bytes else 999.0
        f[11] = abs(flow.fwd_bytes - flow.bwd_bytes)

        # ── Packet size features ───────────────────────────────────────────────
        lengths = flow.pkt_lengths if flow.pkt_lengths else [flow.avg_pkt_len]
        f[12] = float(flow.min_pkt_len)
        f[13] = float(flow.max_pkt_len)
        f[14] = float(FeatureExtractor._mean(lengths))
        f[15] = float(FeatureExtractor._std(lengths))
        mean_len = f[14] if f[14] > 0 else 1.0
        f[16] = f[15] / mean_len                        # CV
        sorted_lens = sorted(lengths)
        f[17] = float(FeatureExtractor._percentile(sorted_lens, 25))
        f[18] = float(FeatureExtractor._percentile(sorted_lens, 50))
        f[19] = float(FeatureExtractor._percentile(sorted_lens, 75))
        f[20] = float(FeatureExtractor._percentile(sorted_lens, 90))

        # ── Timing features ────────────────────────────────────────────────────
        iats = flow.iat_list if flow.iat_list else [0.0]
        f[21] = float(dur)
        f[22] = float(FeatureExtractor._mean(iats))
        f[23] = float(FeatureExtractor._std(iats))
        f[24] = float(min(iats))
        f[25] = float(max(iats))
        mean_iat = f[22] if f[22] > 0 else 1e-9
        f[26] = f[23] / mean_iat                        # IAT CV

        # ── TCP flag features ──────────────────────────────────────────────────
        f[27] = float(flow.syn_count)
        f[28] = float(flow.fin_count)
        f[29] = float(flow.rst_count)
        f[30] = 1.0 if flow.syn_count > 0 else 0.0
        f[31] = 1.0 if flow.rst_count > 0 else 0.0
        f[32] = flow.syn_count / (flow.fin_count + 1)

        # ── Port / service features ────────────────────────────────────────────
        f[33] = float(flow.dst_port)
        f[34] = float(flow.src_port)
        f[35] = 1.0 if flow.dst_port < 1024 else 0.0
        f[36] = 1.0 if flow.dst_port == 443 else 0.0
        f[37] = 1.0 if flow.dst_port == 80  else 0.0
        f[38] = 1.0 if flow.dst_port == 53  else 0.0
        f[39] = 1.0 if flow.src_port >= 49152 else 0.0

        # ── Protocol features ──────────────────────────────────────────────────
        f[40] = 1.0 if flow.protocol == 6  else 0.0
        f[41] = 1.0 if flow.protocol == 17 else 0.0

        # ── Application layer features ─────────────────────────────────────────
        f[42] = 1.0 if flow.sni       else 0.0
        f[43] = 1.0 if flow.http_host else 0.0
        f[44] = 1.0 if flow.dns_query else 0.0

        # ── JA3 (if TLS payload provided) ──────────────────────────────────────
        ja3_str  = ""
        ja3_hash = ""
        if tls_payload:
            result = JA3Fingerprinter.compute(tls_payload)
            if result:
                ja3_str, ja3_hash = result

        flow_id = (f"{flow.src_ip}:{flow.src_port}->"
                   f"{flow.dst_ip}:{flow.dst_port}/{flow.protocol_name}")

        return FeatureVector(
            flow_id=flow_id,
            features=f,
            label=flow.threat_label or flow.app_type.value,
            ja3=ja3_str,
            ja3_hash=ja3_hash,
        )

    @staticmethod
    def extract_batch(flows: List[Flow]) -> List[FeatureVector]:
        """Extract features for a list of flows."""
        return [FeatureExtractor.extract(f) for f in flows]

    # ── Statistical helpers ────────────────────────────────────────────────────

    @staticmethod
    def _mean(data: list) -> float:
        return sum(data) / len(data) if data else 0.0

    @staticmethod
    def _std(data: list) -> float:
        if len(data) < 2:
            return 0.0
        try:
            return statistics.stdev(data)
        except Exception:
            return 0.0

    @staticmethod
    def _percentile(sorted_data: list, p: float) -> float:
        """Compute p-th percentile from sorted list."""
        if not sorted_data:
            return 0.0
        idx = (len(sorted_data) - 1) * p / 100
        lo  = int(idx)
        hi  = min(lo + 1, len(sorted_data) - 1)
        frac = idx - lo
        return sorted_data[lo] * (1 - frac) + sorted_data[hi] * frac