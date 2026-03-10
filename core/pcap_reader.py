"""
core/pcap-reader.py
-------------------
Pure-Python PCAP reader (no external libs required).

PCAP File Format:
  [Global Header 24 bytes]
  [Packet Header 16 bytes][Packet Data N bytes]
  [Packet Header 16 bytes][Packet Data N bytes]
  ...

Global Header layout (24 bytes):
  magic_number  : 4 bytes  (0xa1b2c3d4 = LE, 0xd4c3b2a1 = BE)
  version_major : 2 bytes
  version_minor : 2 bytes
  thiszone      : 4 bytes  (GMT offset, usually 0)
  sigfigs       : 4 bytes  (timestamp accuracy, usually 0)
  snaplen       : 4 bytes  (max bytes per packet)
  network       : 4 bytes  (link-layer type: 1=Ethernet)

Packet Header layout (16 bytes):
  ts_sec   : 4 bytes  (timestamp seconds)
  ts_usec  : 4 bytes  (timestamp microseconds)
  incl_len : 4 bytes  (bytes saved in file)
  orig_len : 4 bytes  (original packet length on wire)
"""
import struct
import os
from dataclasses import dataclass, field
from typing import Iterator, Optional



# ── Data Structures ──────────────────────────────────────────────────────────
@dataclass
class PcapGlobalHeader:
    magic_number: int
    version_major: int
    version_minor: int
    thiszone: int
    sigfigs: int
    snaplen: int
    network: int             # 1=Ethernet(most-common)   
    byte_order: str="little" # 'little' or 'big'

@dataclass
class RawPacket:
    """A single raw packet as read from the PCAP file."""
    ts_sec: int       # Unix timestamp (seconds)
    ts_usec: int      # Microseconds portion
    orig_len: int   # Original length of wire 
    data: bytes     # Captured bytes (may be truncated to snaplen)
    index: int=0    # Packet number (1-based)

    @property
    def timestamp(self) -> float:
        """Full timestamp as float (seconds.microseconds)."""
        return self.ts_sec + self.ts_usec / 1_000_000

    @property
    def captured_len(self) -> int:
        return len(self.data)

    def __repr__(self) -> str:
        return (f"RawPacket(#{self.index} ts={self.timestamp:.6f} "
                f"cap={self.captured_len} orig={self.orig_len})")
    
# ── PcapReader ───────────────────────────────────────────────────────
 
class PcapReader:
    """
    Reads a .pcap file and yields RawPacket objects one at a time.

    Usage:
        reader = PcapReader("capture.pcap")
        for pkt in reader:
            print(pkt)

    Or manually:
        reader.open("capture.pcap")
        while True:
            pkt = reader.read_next()
            if pkt is None:
                break
            process(pkt)
        reader.close()
    """
    GLOBAL_HEADER_SIZE = 24
    PACKET_HEADER_SIZE = 16

    # Link-layer type descriptions
    LINK_TYPES = {
        1: "Ethernet",
        0: "BSD loopback",
        101: "Raw IP",
        113: "Linux cooked",
    }

    def __init__(self, filepath: Optional[str] = None):
        self.fp = None
        self.header :Optional[PcapGlobalHeader] = None
        self._fmt_prefix = "<"  # little-endian by default
        if filepath:
            self.open(filepath)

    # ── Public API ────────────────────────────────────────────────────────────

    def open(self, filepath: str) -> None:
        """Open a PCAP file and parse the global header."""
        if not os.path.isfile(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
        
        self._fp = open(filepath, "rb")
        self._packet_count = 0
        self._parse_global_header()
    
    def close(self) -> None:
        """Close the file  handle."""
        if self._fp:
            self._fp.close()
            self._fp = None
    def read_next(self) -> Optional[RawPacket]:
        """
        Read and return the next packet, or None if EOF.
        """
        if self._fp is None:
            raise RuntimeError("File not open. Call open() first.")

        # Read 16-byte packet header
        raw_hdr = self._fp.read(self.PACKET_HEADER_SIZE)
        if len(raw_hdr) < self.PACKET_HEADER_SIZE:
            return None     # EOF

        ts_sec, ts_usec, incl_len, orig_len = struct.unpack(
            f"{self._fmt_prefix}IIII", raw_hdr
        )

        # Read packet data
        data = self._fp.read(incl_len)
        if len(data) < incl_len:
            return None     # Truncated file

        self._packet_count += 1
        return RawPacket(
            ts_sec=ts_sec,
            ts_usec=ts_usec,
            orig_len=orig_len,
            data=data,
            index=self._packet_count,
        )

    def read_all(self) -> list[RawPacket]:
        """Read all packets into a list (use for small files)."""
        return list(self)

    @property
    def packet_count(self) -> int:
        return self._packet_count

    @property
    def link_type_name(self) -> str:
        if self.header is None:
            return "Unknown"
        return self.LINK_TYPES.get(self.header.network, f"Type {self.header.network}")

    def info(self) -> str:
        """Return a human-readable summary of the PCAP file."""
        if self.header is None:
            return "No file open."
        return (
            f"PCAP v{self.header.version_major}.{self.header.version_minor}  "
            f"Link: {self.link_type_name}  "
            f"Snaplen: {self.header.snaplen}  "
            f"ByteOrder: {self.header.byte_order}-endian"
        )
    
   # ── Iterator Support ──────────────────────────────────────────────────────

    def __iter__(self) -> Iterator[RawPacket]:
        return self

    def __next__(self) -> RawPacket:
        pkt = self.read_next()
        if pkt is None:
            raise StopIteration
        return pkt

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    # ── Private Helpers ───────────────────────────────────────────────────────

    def _parse_global_header(self) -> None:
        raw = self._fp.read(self.GLOBAL_HEADER_SIZE)
        if len(raw) < self.GLOBAL_HEADER_SIZE:
            raise ValueError("File too small to be a valid PCAP.")

        # Detect byte order from magic number
        magic = struct.unpack("<I", raw[:4])[0]
        if magic == 0xa1b2c3d4:
            fmt = "<"   # little-endian (most common)
            byte_order = "little"
        elif magic == 0xd4c3b2a1:
            fmt = ">"   # big-endian
            byte_order = "big"
        else:
            raise ValueError(
                f"Invalid PCAP magic: 0x{magic:08x}. "
                "This may not be a standard PCAP file."
            )

        self._fmt_prefix = fmt
        v_maj, v_min, zone, sigfigs, snaplen, network = struct.unpack(
            f"{fmt}HHiIII", raw[4:]
        )
        self.header = PcapGlobalHeader(
            magic_number=magic,
            version_major=v_maj,
            version_minor=v_min,
            thiszone=zone,
            sigfigs=sigfigs,
            snaplen=snaplen,
            network=network,
            byte_order=byte_order,
        )


# ── PcapWriter ────────────────────────────────────────────────────────────────

class PcapWriter:
    """
    Writes RawPacket objects to a .pcap file.

    Usage:
        with PcapWriter("output.pcap") as w:
            for pkt in filtered_packets:
                w.write(pkt)
    """

    def __init__(self, filepath: str, snaplen: int = 65535, network: int = 1):
        self._filepath = filepath
        self._snaplen = snaplen
        self._network = network
        self._fp = None
        self._count = 0

    def open(self) -> None:
        self._fp = open(self._filepath, "wb")
        self._write_global_header()

    def write(self, pkt: RawPacket) -> None:
        """Write a single packet to the output file."""
        if self._fp is None:
            raise RuntimeError("Writer not open.")
        data = pkt.data[:self._snaplen]
        hdr = struct.pack("<IIII",
                          pkt.ts_sec, pkt.ts_usec,
                          len(data), pkt.orig_len)
        self._fp.write(hdr)
        self._fp.write(data)
        self._count += 1

    def close(self) -> None:
        if self._fp:
            self._fp.close()
            self._fp = None

    @property
    def packets_written(self) -> int:
        return self._count

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *_):
        self.close()

    def _write_global_header(self) -> None:
        hdr = struct.pack("<IHHiIII",
                          0xa1b2c3d4,   # magic (little-endian)
                          2, 4,          # version 2.4
                          0,             # timezone offset
                          0,             # timestamp accuracy
                          self._snaplen,
                          self._network)
        self._fp.write(hdr)