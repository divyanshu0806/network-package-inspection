"""
ml/feature_pipeline.py
-----------------------
End-to-end feature extraction pipeline.

Wires together:
  FlowTracker → FeatureExtractor → AnomalyDetector →
  BeaconingDetector → PortScanDetector → CSV/JSON export

This is the "glue" module that Phase 3 (ML) and Phase 4 (AI) both consume.

Output formats:
  1. flows.csv      — one row per flow with all 45 ML features + metadata
  2. anomalies.csv  — only anomalous flows
  3. summary.json   — high-level stats, top threats, app breakdown
  4. ja3_report.csv — JA3 fingerprint frequency table
"""

import csv
import json
import os
import time
from typing import List, Optional, Dict, Any
from collections import Counter, defaultdict

from core.flow_tracker    import FlowTracker, Flow
from core.feature_extractor import FeatureExtractor, FeatureVector, FEATURE_NAMES
from ml.anomaly_detector  import IsolationForest
from ml.beaconing_detector import BeaconingDetector, BeaconResult
from ml.port_scan_detector import PortScanDetector, ExfiltrationDetector, ScanResult, ExfilResult


class FeaturePipeline:
    """
    Complete feature extraction and anomaly detection pipeline.

    Usage:
        pipeline = FeaturePipeline()
        pipeline.run(flow_tracker)
        pipeline.export("output_dir/")
        report = pipeline.summary()
    """

    def __init__(self,
                 n_trees:       int   = 100,
                 contamination: float = 0.10,
                 beacon_cv:     float = 0.40,
                 scan_min_ports: int  = 10):
        self.n_trees        = n_trees
        self.contamination  = contamination
        self.beacon_cv      = beacon_cv
        self.scan_min_ports = scan_min_ports

        # Results populated by run()
        self.feature_vectors: List[FeatureVector] = []
        self.beacon_results:  List[BeaconResult]  = []
        self.scan_results:    List[ScanResult]    = []
        self.exfil_results:   List[ExfilResult]   = []
        self.forest:          Optional[IsolationForest] = None
        self._ran = False

    def run(self, tracker: FlowTracker) -> "FeaturePipeline":
        """
        Run the full pipeline on a FlowTracker.
        Call after all packets have been processed.
        """
        flows = tracker.all_flows()
        if not flows:
            print("[Pipeline] No flows to process.")
            return self

        print(f"\n[Pipeline] Processing {len(flows)} flows...")

        # ── Step 1: Feature extraction ─────────────────────────────────────────
        print("[Pipeline] Extracting features...")
        self.feature_vectors = FeatureExtractor.extract_batch(flows)

        # ── Step 2: Isolation Forest anomaly detection ─────────────────────────
        if len(self.feature_vectors) >= 4:
            print(f"[Pipeline] Training Isolation Forest "
                  f"({self.n_trees} trees, contamination={self.contamination:.0%})...")
            self.forest = IsolationForest(
                n_trees=self.n_trees,
                contamination=self.contamination,
            )
            self.forest.fit(self.feature_vectors)
            self.forest.annotate(self.feature_vectors)

            # Write scores back to Flow objects
            for fv, flow in zip(self.feature_vectors, flows):
                flow.anomaly_score = fv.anomaly_score
                flow.threat_label  = "anomaly" if fv.is_anomaly else ""

            n_anomalies = sum(1 for fv in self.feature_vectors if fv.is_anomaly)
            print(f"[Pipeline] Anomaly detection: threshold={self.forest.threshold:.3f} | "
                  f"{n_anomalies}/{len(flows)} flagged")
        else:
            print("[Pipeline] Too few flows for Isolation Forest (need ≥ 4).")

        # ── Step 3: Beaconing detection ────────────────────────────────────────
        print("[Pipeline] Running beaconing detector...")
        beacon_det = BeaconingDetector(cv_threshold=self.beacon_cv)
        self.beacon_results = beacon_det.analyze(flows)
        if self.beacon_results:
            n_beacons = sum(1 for r in self.beacon_results if r.is_beacon)
            print(f"[Pipeline] Beaconing: {n_beacons} potential beacons found")

        # ── Step 4: Port scan detection ────────────────────────────────────────
        print("[Pipeline] Running port scan detector...")
        scan_det = PortScanDetector(min_ports=self.scan_min_ports)
        self.scan_results = scan_det.analyze(flows)
        if self.scan_results:
            n_scanners = sum(1 for r in self.scan_results if r.is_scanner)
            print(f"[Pipeline] Port scans: {n_scanners} scanners found")

        # ── Step 5: Exfiltration detection ─────────────────────────────────────
        print("[Pipeline] Running exfiltration detector...")
        exfil_det = ExfiltrationDetector()
        self.exfil_results = exfil_det.analyze(flows)
        if self.exfil_results:
            n_exfil = sum(1 for r in self.exfil_results if r.is_exfil)
            print(f"[Pipeline] Exfiltration: {n_exfil} suspicious flows found")

        self._ran = True
        return self

    def export(self, output_dir: str = ".") -> Dict[str, str]:
        """
        Write all outputs to disk.
        Returns dict of {name: filepath}.
        """
        os.makedirs(output_dir, exist_ok=True)
        paths = {}

        # ── flows.csv ──────────────────────────────────────────────────────────
        flows_path = os.path.join(output_dir, "flows.csv")
        self._write_flows_csv(flows_path)
        paths["flows_csv"] = flows_path
        print(f"[Export] flows.csv → {flows_path} ({len(self.feature_vectors)} rows)")

        # ── anomalies.csv ──────────────────────────────────────────────────────
        anomalies = [fv for fv in self.feature_vectors if fv.is_anomaly]
        if anomalies:
            anom_path = os.path.join(output_dir, "anomalies.csv")
            self._write_vectors_csv(anomalies, anom_path)
            paths["anomalies_csv"] = anom_path
            print(f"[Export] anomalies.csv → {anom_path} ({len(anomalies)} rows)")

        # ── ja3_report.csv ─────────────────────────────────────────────────────
        ja3_flows = [fv for fv in self.feature_vectors if fv.ja3_hash]
        if ja3_flows:
            ja3_path = os.path.join(output_dir, "ja3_report.csv")
            self._write_ja3_csv(ja3_flows, ja3_path)
            paths["ja3_csv"] = ja3_path

        # ── summary.json ───────────────────────────────────────────────────────
        summary_path = os.path.join(output_dir, "summary.json")
        summary = self.summary()
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        paths["summary_json"] = summary_path
        print(f"[Export] summary.json → {summary_path}")

        return paths

    def summary(self) -> Dict[str, Any]:
        """Return a comprehensive summary dict for reporting/AI."""
        fvs = self.feature_vectors

        total = len(fvs)
        n_anomaly  = sum(1 for fv in fvs if fv.is_anomaly)
        n_beacons  = sum(1 for r in self.beacon_results if r.is_beacon)
        n_scanners = sum(1 for r in self.scan_results if r.is_scanner)
        n_exfil    = sum(1 for r in self.exfil_results if r.is_exfil)

        # Top labels
        label_counts = Counter(fv.label for fv in fvs)

        # Top anomalies (sorted by score)
        top_anomalies = sorted(
            [fv for fv in fvs if fv.is_anomaly],
            key=lambda x: -x.anomaly_score
        )[:10]

        # JA3 frequency
        ja3_counts = Counter(fv.ja3_hash for fv in fvs if fv.ja3_hash)

        return {
            "generated_at":   time.strftime("%Y-%m-%d %Human:%M:%S"),
            "total_flows":    total,
            "anomaly_count":  n_anomaly,
            "anomaly_rate":   round(n_anomaly / total, 4) if total else 0,
            "beacon_count":   n_beacons,
            "scanner_count":  n_scanners,
            "exfil_count":    n_exfil,
            "model_threshold": self.forest.threshold if self.forest else None,
            "label_distribution": dict(label_counts.most_common(15)),
            "top_anomalies": [
                {
                    "flow_id":       fv.flow_id,
                    "anomaly_score": fv.anomaly_score,
                    "label":         fv.label,
                    "pkt_count":     int(fv["pkt_count"]),
                    "byte_count":    int(fv["byte_count"]),
                    "dst_port":      int(fv["dst_port"]),
                }
                for fv in top_anomalies
            ],
            "beacons": [
                {
                    "src_ip":    r.src_ip,
                    "dst_ip":    r.dst_ip,
                    "dst_port":  r.dst_port,
                    "interval":  round(r.estimated_interval, 2),
                    "cv":        round(r.iat_cv, 4),
                    "score":     r.beacon_score,
                    "connections": r.connection_count,
                }
                for r in self.beacon_results if r.is_beacon
            ],
            "port_scans": [
                {
                    "src_ip":      r.src_ip,
                    "scan_type":   r.scan_type,
                    "port_count":  r.port_count,
                    "target_count": r.target_count,
                    "confidence":  r.confidence,
                }
                for r in self.scan_results if r.is_scanner
            ],
            "exfiltration": [
                {
                    "src_ip":    r.src_ip,
                    "dst_ip":    r.dst_ip,
                    "dst_port":  r.dst_port,
                    "bytes":     r.byte_count,
                    "rate_kbps": round(r.byte_rate / 1024, 1),
                    "signals":   r.signals,
                    "confidence": r.confidence,
                }
                for r in self.exfil_results if r.is_exfil
            ],
            "ja3_fingerprints": [
                {"hash": h, "count": c}
                for h, c in ja3_counts.most_common(20)
            ],
        }

    def print_report(self) -> None:
        """Print a human-readable Phase 2 analysis report to stdout."""
        s = self.summary()

        C_RESET  = "\033[0m"
        C_BOLD   = "\033[1m"
        C_RED    = "\033[91m"
        C_YELLOW = "\033[93m"
        C_CYAN   = "\033[96m"
        C_GREEN  = "\033[92m"
        C_DIM    = "\033[2m"

        print(f"""
{C_BOLD}{C_CYAN}
╔══════════════════════════════════════════════════════════════════╗
║               🧠 PHASE 2 — ML FEATURE ANALYSIS REPORT           ║
╚══════════════════════════════════════════════════════════════════╝{C_RESET}

{C_BOLD}Threat Overview{C_RESET}
  Total Flows     : {s['total_flows']}
  Anomalies       : {C_RED}{s['anomaly_count']} ({s['anomaly_rate']:.1%}){C_RESET}
  Beacon suspects : {C_YELLOW}{s['beacon_count']}{C_RESET}
  Port scanners   : {C_YELLOW}{s['scanner_count']}{C_RESET}
  Exfil suspects  : {C_RED}{s['exfil_count']}{C_RESET}
""")

        if s["top_anomalies"]:
            print(f"{C_BOLD}Top Anomalous Flows{C_RESET}")
            for a in s["top_anomalies"]:
                print(f"  {C_RED}[{a['anomaly_score']:.3f}]{C_RESET}  "
                      f"{a['flow_id']}  "
                      f"{a['pkt_count']}pkts  {a['byte_count']//1024}KB")

        if s["beacons"]:
            print(f"\n{C_BOLD}Beaconing Suspects{C_RESET}")
            for b in s["beacons"]:
                mins = b['interval'] / 60
                interval_str = f"{mins:.1f}min" if mins >= 1 else f"{b['interval']:.1f}s"
                print(f"  {C_YELLOW}⏱  {b['src_ip']} → {b['dst_ip']}:{b['dst_port']}{C_RESET}"
                      f"  interval≈{interval_str}  CV={b['cv']:.3f}"
                      f"  score={b['score']:.2f}  ({b['connections']} connections)")

        if s["port_scans"]:
            print(f"\n{C_BOLD}Port Scan Activity{C_RESET}")
            for sc in s["port_scans"]:
                print(f"  {C_YELLOW}🔍 {sc['src_ip']}{C_RESET}"
                      f"  {sc['scan_type'].upper()}"
                      f"  {sc['port_count']} ports"
                      f"  {sc['target_count']} hosts"
                      f"  conf={sc['confidence']:.2f}")

        if s["exfiltration"]:
            print(f"\n{C_BOLD}Exfiltration Suspects{C_RESET}")
            for e in s["exfiltration"]:
                mb = e['bytes'] / (1024*1024)
                print(f"  {C_RED}🚨 {e['src_ip']} → {e['dst_ip']}:{e['dst_port']}{C_RESET}"
                      f"  {mb:.2f}MB @ {e['rate_kbps']:.1f}KB/s"
                      f"  conf={e['confidence']:.2f}"
                      f"  signals={e['signals']}")

        if s["ja3_fingerprints"]:
            print(f"\n{C_BOLD}JA3 TLS Fingerprints{C_RESET}")
            for j in s["ja3_fingerprints"][:5]:
                print(f"  {j['hash']}  (seen {j['count']}x)")

        print(f"\n{C_DIM}Phase 3 next: Claude AI threat analysis and incident reports.{C_RESET}\n")

    # ── Private write helpers ──────────────────────────────────────────────────

    def _write_flows_csv(self, path: str) -> None:
        if not self.feature_vectors:
            return
        fieldnames = (["flow_id", "label", "anomaly_score", "is_anomaly",
                       "ja3", "ja3_hash"] + FEATURE_NAMES)
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for fv in self.feature_vectors:
                w.writerow(fv.to_dict())

    def _write_vectors_csv(self, vectors: List[FeatureVector], path: str) -> None:
        if not vectors:
            return
        fieldnames = (["flow_id", "label", "anomaly_score", "is_anomaly",
                       "ja3", "ja3_hash"] + FEATURE_NAMES)
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for fv in vectors:
                w.writerow(fv.to_dict())

    def _write_ja3_csv(self, vectors: List[FeatureVector], path: str) -> None:
        ja3_map: Dict[str, Dict] = {}
        for fv in vectors:
            if fv.ja3_hash not in ja3_map:
                ja3_map[fv.ja3_hash] = {
                    "ja3_hash": fv.ja3_hash,
                    "ja3_string": fv.ja3,
                    "count": 0,
                    "flows": [],
                }
            ja3_map[fv.ja3_hash]["count"] += 1
            if len(ja3_map[fv.ja3_hash]["flows"]) < 3:
                ja3_map[fv.ja3_hash]["flows"].append(fv.flow_id)

        rows = sorted(ja3_map.values(), key=lambda x: -x["count"])
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["ja3_hash", "ja3_string", "count", "flows"])
            w.writeheader()
            for row in rows:
                row["flows"] = " | ".join(row["flows"])
                w.writerow(row)