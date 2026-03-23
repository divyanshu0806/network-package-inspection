"""
main.py
-------
DPI AI Engine — Phase 1 Entry Point

Orchestrates the full packet inspection pipeline:
  1. Read PCAP file
  2. Parse each packet (Ethernet → IP → TCP/UDP)
  3. Extract domain names via DPI (TLS SNI, HTTP Host, DNS)
  4. Track sessions in flow table
  5. Apply basic block rules
  6. Write filtered output PCAP
  7. Print analysis report

Usage:
  python3 main.py <input.pcap> [output.pcap] [--block-domain <domain>] [--block-ip <ip>]

Examples:
  python3 main.py test_traffic.pcap
  python3 main.py test_traffic.pcap filtered.pcap
  python3 main.py test_traffic.pcap out.pcap --block-domain tiktok --block-domain facebook
  python3 main.py test_traffic.pcap out.pcap --block-ip 192.168.1.99
"""

import sys
import os
import argparse
import time
from collections import Counter

# ── Add project root to path ──────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

from core.pcap_reader    import PcapReader, PcapWriter
from core.packet_parser  import PacketParser
from core.sni_extractor  import DPIInspector
from core.flow_tracker   import FlowTracker, FlowState
from ml.feature_pipeline import FeaturePipeline


# ── ANSI Colors (works on all modern terminals) ───────────────────────────────
class C:
    RESET  = "\033[0m"
    BOLD   = "\033[1m"
    RED    = "\033[91m"
    GREEN  = "\033[92m"
    YELLOW = "\033[93m"
    BLUE   = "\033[94m"
    CYAN   = "\033[96m"
    WHITE  = "\033[97m"
    DIM    = "\033[2m"
    BG_RED = "\033[41m"


def banner():
    print(f"""
{C.BOLD}{C.CYAN}
╔══════════════════════════════════════════════════════════════════╗
║        🛡️  AI-Powered Deep Packet Inspection Engine              ║
║                    Phase 1 — Core Engine                         ║
╚══════════════════════════════════════════════════════════════════╝
{C.RESET}""")


def bar(value: int, total: int, width: int = 20) -> str:
    """ASCII progress bar."""
    if total == 0:
        return " " * width
    filled = int(width * value / total)
    return "█" * filled + "░" * (width - filled)


# ── Rule Engine (basic, Phase 4 will use AI) ──────────────────────────────────

class SimpleRuleEngine:
    """
    Basic block/allow rules by IP, domain, or app name.
    Phase 4 will replace/augment this with AI-driven rules.
    """

    def __init__(self):
        self.blocked_ips:     set = set()
        self.blocked_domains: set = set()   # substring match
        self.blocked_apps:    set = set()   # AppType value strings

    def add_blocked_ip(self, ip: str):
        self.blocked_ips.add(ip.strip())
        print(f"  {C.YELLOW}[Rule]{C.RESET} Block IP: {ip}")

    def add_blocked_domain(self, domain: str):
        self.blocked_domains.add(domain.strip().lower())
        print(f"  {C.YELLOW}[Rule]{C.RESET} Block domain: {domain}")

    def add_blocked_app(self, app: str):
        self.blocked_apps.add(app.strip().lower())
        print(f"  {C.YELLOW}[Rule]{C.RESET} Block app: {app}")

    def should_block(self, src_ip: str, domain: str, app_type: str) -> tuple[bool, str]:
        """Returns (blocked, reason)."""
        if src_ip in self.blocked_ips:
            return True, f"IP {src_ip} is blocked"

        if domain:
            for bd in self.blocked_domains:
                if bd in domain.lower():
                    return True, f"Domain '{domain}' matches block rule '{bd}'"

        if app_type.lower() in self.blocked_apps:
            return True, f"App '{app_type}' is blocked"

        return False, ""


# ── Main Pipeline ─────────────────────────────────────────────────────────────

def run(input_pcap: str,
        output_pcap: str = None,
        blocked_domains: list = None,
        blocked_ips: list = None,
        blocked_apps: list = None,
        verbose: bool = False,
        export_dir: str = None) -> dict:

    banner()

    # ── Setup ──────────────────────────────────────────────────────────────────
    rules   = SimpleRuleEngine()
    tracker = FlowTracker()
    parser  = PacketParser()

    for domain in (blocked_domains or []):
        rules.add_blocked_domain(domain)
    for ip in (blocked_ips or []):
        rules.add_blocked_ip(ip)
    for app in (blocked_apps or []):
        rules.add_blocked_app(app)

    # ── Open files ────────────────────────────────────────────────────────────
    reader = PcapReader(input_pcap)
    print(f"\n{C.BOLD}[Reader]{C.RESET} {reader.info()}")
    print(f"         File: {input_pcap}")

    writer = None
    if output_pcap:
        writer = PcapWriter(output_pcap)
        writer.open()
        print(f"         Output: {output_pcap}")

    # ── Counters ───────────────────────────────────────────────────────────────
    total_pkts    = 0
    forwarded     = 0
    dropped       = 0
    dpi_hits      = 0
    parse_errors  = 0
    start_time    = time.time()

    print(f"\n{C.BOLD}[Processing]{C.RESET} Reading packets...\n")

    # ── Main Packet Loop ───────────────────────────────────────────────────────
    for raw_pkt in reader:
        total_pkts += 1

        # ── 1. Parse packet ───────────────────────────────────────────────────
        pkt = PacketParser.parse(raw_pkt)

        if pkt.errors:
            parse_errors += 1
            if verbose:
                print(f"  {C.DIM}[Parse Error #{raw_pkt.index}] {pkt.errors}{C.RESET}")

        if not pkt.src_ip:
            # Non-IP packet (ARP, etc.) → forward as-is
            if writer:
                writer.write(raw_pkt)
            forwarded += 1
            continue

        # ── 2. Flow lookup / creation ─────────────────────────────────────────
        flow = tracker.get_or_create(pkt)
        flow.update(pkt)

        # ── 3. DPI — extract domain if not already known ──────────────────────
        if flow.domain is None and pkt.payload:
            dpi_result = DPIInspector.inspect(
                pkt.payload, pkt.src_port, pkt.dst_port
            )
            if dpi_result:
                dpi_hits += 1
                flow.set_dpi_result(
                    domain=dpi_result.domain,
                    source=dpi_result.source,
                    tls_version=dpi_result.tls_version,
                )
                if verbose:
                    src = dpi_result.source.upper()
                    dom = dpi_result.domain or ""
                    # Skip garbled binary data from false-positive DNS parses
                    if len(dom) > 80 or not all(32 <= ord(c) < 127 for c in dom):
                        dom = "[binary/garbled]"
                    print(f"  {C.GREEN}[DPI/{src}]{C.RESET} "
                          f"#{raw_pkt.index} {pkt.src_ip} → {pkt.dst_ip} : "
                          f"{C.BOLD}{dom}{C.RESET}")

        # ── 4. Rule check ─────────────────────────────────────────────────────
        if not flow.blocked:
            blocked, reason = rules.should_block(
                pkt.src_ip,
                flow.domain or "",
                flow.app_type.value,
            )
            if blocked:
                flow.blocked = True
                flow.block_reason = reason
                flow.state = FlowState.BLOCKED
                print(f"  {C.RED}[BLOCKED]{C.RESET} Flow {pkt.src_ip}→{pkt.dst_ip} "
                      f"| {reason}")

        # ── 5. Forward or drop ────────────────────────────────────────────────
        if flow.blocked:
            dropped += 1
        else:
            if writer:
                writer.write(raw_pkt)
            forwarded += 1

        # Progress indicator every 100 packets
        if total_pkts % 100 == 0:
            print(f"  ... {total_pkts} packets processed", end="\r")

    # ── Cleanup ────────────────────────────────────────────────────────────────
    reader.close()
    if writer:
        writer.close()

    elapsed = time.time() - start_time

    # ── Report ─────────────────────────────────────────────────────────────────
    _print_report(tracker, total_pkts, forwarded, dropped,
                  dpi_hits, parse_errors, elapsed, output_pcap)

    # ── Phase 2: ML Feature Extraction + Anomaly Detection ───────────────────
    pipeline = FeaturePipeline()
    pipeline.run(tracker)
    pipeline.print_report()

    # Export to disk if requested
    if export_dir:
        paths = pipeline.export(export_dir)
        print(f"\n{C.GREEN}[Export] Files written to: {export_dir}{C.RESET}")
        for name, path in paths.items():
            print(f"  {name}: {path}")

    return {
        "total_packets": total_pkts,
        "forwarded":     forwarded,
        "dropped":       dropped,
        "dpi_hits":      dpi_hits,
        "flows":         tracker.flow_count,
        "elapsed":       elapsed,
        "anomalies":     sum(1 for fv in pipeline.feature_vectors if fv.is_anomaly),
        "beacons":       sum(1 for r in pipeline.beacon_results if r.is_beacon),
        "scanners":      sum(1 for r in pipeline.scan_results if r.is_scanner),
    }


def _print_report(tracker: FlowTracker, total: int, forwarded: int,
                  dropped: int, dpi_hits: int, errors: int,
                  elapsed: float, output_path: str):
    """Print the final analysis report."""

    flows   = tracker.all_flows()
    stats   = tracker.stats_summary()
    rate    = total / elapsed if elapsed > 0 else 0

    print(f"""
{C.BOLD}{C.CYAN}
╔══════════════════════════════════════════════════════════════════╗
║                     📊 ANALYSIS REPORT                           ║
╚══════════════════════════════════════════════════════════════════╝{C.RESET}

{C.BOLD}Packet Statistics{C.RESET}
  Total Packets  : {C.WHITE}{total:>8,}{C.RESET}
  Forwarded      : {C.GREEN}{forwarded:>8,}{C.RESET}
  Dropped        : {C.RED}{dropped:>8,}{C.RESET}
  Parse Errors   : {C.DIM}{errors:>8,}{C.RESET}
  DPI Hits       : {C.CYAN}{dpi_hits:>8,}{C.RESET}
  Processing Rate: {rate:>8.0f} pkt/s  ({elapsed:.2f}s total)

{C.BOLD}Flow Statistics{C.RESET}
  Total Flows    : {C.WHITE}{tracker.flow_count:>8,}{C.RESET}
  Total Created  : {tracker.total_created:>8,}
  Blocked Flows  : {C.RED}{len(tracker.blocked_flows()):>8,}{C.RESET}
""")

    # ── App Breakdown ──────────────────────────────────────────────────────────
    if stats.get("app_breakdown"):
        print(f"{C.BOLD}Application Breakdown  (by flow count){C.RESET}")
        app_data = stats["app_breakdown"]
        max_count = max(app_data.values()) if app_data else 1
        for app, count in sorted(app_data.items(), key=lambda x: -x[1]):
            pct  = count / tracker.flow_count * 100 if tracker.flow_count else 0
            b    = bar(count, max_count, 20)
            blocked_flows = [f for f in flows if f.app_type.value == app and f.blocked]
            block_str = f"  {C.RED}[BLOCKED: {len(blocked_flows)}]{C.RESET}" if blocked_flows else ""
            print(f"  {app:<20} {count:>4}  {pct:5.1f}%  {C.BLUE}{b}{C.RESET}{block_str}")

    # ── Top Domains ────────────────────────────────────────────────────────────
    if stats.get("top_domains"):
        print(f"\n{C.BOLD}Top Domains / SNIs{C.RESET}")
        for domain, count in stats["top_domains"].items():
            blocked = any(f.blocked for f in flows if f.domain == domain)
            block_str = f"  {C.RED}⛔ BLOCKED{C.RESET}" if blocked else ""
            print(f"  {domain:<40} {count:>3} flows{block_str}")

    # ── Protocol Breakdown ────────────────────────────────────────────────────
    if stats.get("protocol_breakdown"):
        print(f"\n{C.BOLD}Protocol Breakdown{C.RESET}")
        for proto, count in stats["protocol_breakdown"].items():
            pct = count / tracker.flow_count * 100 if tracker.flow_count else 0
            print(f"  {proto:<8} {count:>4} flows  {pct:.1f}%")

    # ── Flagged / Suspicious Flows ────────────────────────────────────────────
    high_vol = [f for f in flows
                if f.pkt_count > 30 or f.byte_count > 50_000]
    if high_vol:
        print(f"\n{C.BOLD}{C.YELLOW}⚠️  High-Volume Flows (potential anomalies — for ML in Phase 3){C.RESET}")
        for f in sorted(high_vol, key=lambda x: -x.byte_count)[:5]:
            print(f"  {f.src_ip}:{f.src_port} → {f.dst_ip}:{f.dst_port}"
                  f"  {f.pkt_count}pkts  {f.byte_count/1024:.1f}KB"
                  f"  {f.protocol_name}"
                  + (f"  [{f.domain}]" if f.domain else ""))

    if output_path:
        print(f"\n{C.GREEN}✅ Filtered output written to: {output_path}{C.RESET}")

    print(f"\n{C.DIM}Phase 2 next: Flow feature extraction for ML training.{C.RESET}\n")


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="DPI AI Engine — Phase 1 Core",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    ap.add_argument("input",  help="Input PCAP file")
    ap.add_argument("output", nargs="?", help="Output PCAP file (filtered)")
    ap.add_argument("--block-domain", "-d", action="append", default=[],
                    metavar="DOMAIN", help="Block flows matching this domain (repeatable)")
    ap.add_argument("--block-ip", "-i", action="append", default=[],
                    metavar="IP", help="Block flows from this source IP (repeatable)")
    ap.add_argument("--block-app", "-a", action="append", default=[],
                    metavar="APP", help="Block app by name (e.g. TikTok, YouTube)")
    ap.add_argument("--verbose", "-v", action="store_true",
                    help="Show per-packet DPI results")
    ap.add_argument("--export", "-e", metavar="DIR",
                    help="Export ML features, anomalies, and summary JSON to DIR")

    args = ap.parse_args()

    if not os.path.exists(args.input):
        print(f"{C.RED}Error: File not found: {args.input}{C.RESET}")
        sys.exit(1)

    run(
        input_pcap=args.input,
        output_pcap=args.output,
        blocked_domains=args.block_domain,
        blocked_ips=args.block_ip,
        blocked_apps=args.block_app,
        verbose=args.verbose,
        export_dir=args.export,
    )


if __name__ == "__main__":
    main()