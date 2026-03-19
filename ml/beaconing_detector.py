"""
ml/beaconing_detector.py
------------------------
Detects Command-and-Control (C2) beaconing traffic using statistical analysis
of inter-arrival times (IAT) and flow periodicity.

What is beaconing?
  Malware infected machines "phone home" to a C2 server at regular intervals
  to receive commands or exfiltrate data. These connections are periodic:
    - Fixed interval: every 30 seconds, every 5 minutes, etc.
    - Jittered interval: ±10% random jitter to evade detection
    - Low data volume: small keepalive packets

Why pure statistics (no ML model needed)?
  Beaconing has a distinctive statistical signature:
    - Low IAT coefficient of variation (very regular timing)
    - Many connections to the same destination
    - Small, consistent packet sizes
    - Long duration (keeps running in background)

Detection Method:
  1. Compute IAT coefficient of variation (CV = std/mean)
     - CV < 0.4  = very regular → suspicious
     - CV < 0.15 = extremely regular → almost certainly beaconing
  2. Check minimum connection count (≥ 5 connections to same dest)
  3. Check packet size variance (beacons are typically small + uniform)
  4. Score 0.0–1.0: 1.0 = definitely beaconing

Known beacon intervals:
  - Cobalt Strike default: 60s
  - Metasploit Meterpreter: varies
  - Emotet: 300s
  - Trickbot: 600s
"""

import math
import statistics
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from collections import defaultdict


@dataclass
class BeaconResult:
    """Result of beacon analysis for a single destination."""
    dst_ip:       str
    dst_port:     int
    src_ip:       str
    connection_count: int
    avg_interval: float        # seconds between connections
    iat_cv:       float        # coefficient of variation (lower = more regular)
    beacon_score: float        # 0.0 – 1.0 (1.0 = definitely beaconing)
    is_beacon:    bool
    estimated_interval: float  # most likely beacon period (seconds)
    intervals:    List[float]  # raw IAT values
    verdict:      str = ""

    def __post_init__(self):
        if not self.verdict:
            if self.is_beacon:
                mins = self.estimated_interval / 60
                if mins >= 1:
                    self.verdict = (f"⚠️  BEACON DETECTED — {self.src_ip} → "
                                    f"{self.dst_ip}:{self.dst_port} | "
                                    f"Interval ≈ {mins:.1f}min | "
                                    f"CV={self.iat_cv:.3f} | "
                                    f"Score={self.beacon_score:.2f}")
                else:
                    self.verdict = (f"⚠️  BEACON DETECTED — {self.src_ip} → "
                                    f"{self.dst_ip}:{self.dst_port} | "
                                    f"Interval ≈ {self.estimated_interval:.1f}s | "
                                    f"CV={self.iat_cv:.3f} | "
                                    f"Score={self.beacon_score:.2f}")
            else:
                self.verdict = (f"OK — {self.src_ip} → {self.dst_ip}:{self.dst_port} | "
                                f"CV={self.iat_cv:.3f} | Score={self.beacon_score:.2f}")


class BeaconingDetector:
    """
    Detects C2 beaconing by analyzing flow timing patterns.

    Works on completed flows from the FlowTracker — analyzes
    per-destination IAT patterns to find periodic connections.

    Usage:
        detector = BeaconingDetector()
        results = detector.analyze(flow_tracker.all_flows())
        for r in results:
            if r.is_beacon:
                print(r.verdict)
    """

    # Thresholds (tunable)
    MIN_CONNECTIONS   = 4       # Need at least this many connections to the same dest
    BEACON_CV_THRESH  = 0.40    # IAT CV below this = suspicious
    BEACON_SCORE_THRESH = 0.55  # Score above this = flagged as beacon

    def __init__(self,
                 min_connections: int   = MIN_CONNECTIONS,
                 cv_threshold: float    = BEACON_CV_THRESH,
                 score_threshold: float = BEACON_SCORE_THRESH):
        self.min_connections   = min_connections
        self.cv_threshold      = cv_threshold
        self.score_threshold   = score_threshold

    def analyze(self, flows) -> List[BeaconResult]:
        """
        Analyze a list of Flow objects for beaconing patterns.
        Groups flows by (src_ip, dst_ip, dst_port) and looks for
        repeated periodic connections.
        """
        # Group flows by (src, dst, port) — same beacon target
        groups: Dict[tuple, list] = defaultdict(list)
        for flow in flows:
            key = (flow.src_ip, flow.dst_ip, flow.dst_port)
            groups[key].append(flow)

        results = []
        for (src, dst, dport), group_flows in groups.items():
            if len(group_flows) < self.min_connections:
                continue

            result = self._analyze_group(src, dst, dport, group_flows)
            if result:
                results.append(result)

        # Sort by beacon score descending
        return sorted(results, key=lambda r: -r.beacon_score)

    def analyze_flow_iats(self, src_ip: str, dst_ip: str,
                           dst_port: int, iats: List[float]) -> Optional[BeaconResult]:
        """
        Analyze a single flow's internal IAT list directly.
        Useful when a single long-lived flow shows periodic behavior.
        """
        if len(iats) < self.min_connections:
            return None
        return self._score_iats(src_ip, dst_ip, dst_port, len(iats), iats)

    # ── Private helpers ────────────────────────────────────────────────────────

    def _analyze_group(self, src: str, dst: str, dport: int,
                        flows: list) -> Optional[BeaconResult]:
        """
        Analyze a group of flows from the same src→dst:port.
        Uses the first-seen timestamps as connection times.
        """
        # Sort by first_seen timestamp
        sorted_flows = sorted(flows, key=lambda f: f.first_seen)
        times = [f.first_seen for f in sorted_flows]

        # Compute inter-connection intervals
        if len(times) < 2:
            return None
        iats = [times[i+1] - times[i] for i in range(len(times)-1)]

        return self._score_iats(src, dst, dport, len(flows), iats)

    def _score_iats(self, src: str, dst: str, dport: int,
                     n_connections: int, iats: List[float]) -> BeaconResult:
        """
        Score a list of inter-arrival times for beaconing likelihood.
        Returns a BeaconResult with a 0–1 score.
        """
        if not iats or len(iats) < 1:
            return BeaconResult(
                dst_ip=dst, dst_port=dport, src_ip=src,
                connection_count=n_connections,
                avg_interval=0, iat_cv=999,
                beacon_score=0, is_beacon=False,
                estimated_interval=0, intervals=[]
            )

        mean_iat = statistics.mean(iats)
        std_iat  = statistics.stdev(iats) if len(iats) > 1 else 0.0
        cv       = std_iat / mean_iat if mean_iat > 0 else 999.0

        score = self._compute_score(cv, n_connections, mean_iat, iats)
        is_beacon = score >= self.score_threshold

        # Estimate the most likely beacon interval
        estimated = self._estimate_period(iats)

        return BeaconResult(
            dst_ip=dst,
            dst_port=dport,
            src_ip=src,
            connection_count=n_connections,
            avg_interval=mean_iat,
            iat_cv=cv,
            beacon_score=score,
            is_beacon=is_beacon,
            estimated_interval=estimated,
            intervals=iats[:50],    # keep first 50 for reporting
        )

    def _compute_score(self, cv: float, n: int,
                        mean_iat: float, iats: List[float]) -> float:
        """
        Compute beacon score 0.0–1.0 from multiple signals.

        Signal 1: IAT coefficient of variation
          CV=0.0 → perfect beacon (score=1.0)
          CV=0.4 → threshold (score≈0.5)
          CV>1.0 → random/bursty (score→0.0)

        Signal 2: Connection count bonus
          More connections = more confident

        Signal 3: Interval consistency
          Check if most intervals cluster around the mean

        Signal 4: Interval range (extremes penalize score)
        """
        # Signal 1: CV score (main signal)
        # Score = max(0, 1 - CV/0.5)  → CV=0 → 1.0, CV=0.5 → 0.0
        cv_score = max(0.0, 1.0 - cv / 0.5)

        # Signal 2: Count bonus (asymptotes to 1.0 as n → ∞)
        count_score = 1.0 - math.exp(-0.3 * (n - self.min_connections))
        count_score = max(0.0, min(1.0, count_score))

        # Signal 3: Fraction of intervals within 30% of mean
        near_mean = sum(1 for x in iats
                        if abs(x - mean_iat) / max(mean_iat, 1e-9) < 0.30)
        consistency = near_mean / len(iats)

        # Signal 4: Penalize very short intervals (< 1s) — likely noise not beacon
        if mean_iat < 1.0:
            short_penalty = 0.5
        else:
            short_penalty = 1.0

        # Weighted combination
        score = (0.50 * cv_score +
                 0.20 * count_score +
                 0.30 * consistency) * short_penalty

        return round(min(1.0, max(0.0, score)), 4)

    @staticmethod
    def _estimate_period(iats: List[float]) -> float:
        """
        Estimate the most likely beacon period.
        Uses median (more robust than mean for jittered beacons).
        """
        if not iats:
            return 0.0
        return statistics.median(iats)