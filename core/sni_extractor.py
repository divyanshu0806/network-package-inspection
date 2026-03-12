"""
core/sni_extractor.py
---------------------
Deep Packet Inspection: extract domain names from packet payloads.

Three sources:
  1. TLS SNI      — from TLS ClientHello extension (HTTPS port 443)
  2. HTTP Host    — from plaintext HTTP "Host:" header (port 80)
  3. DNS Query    — from DNS question section (port 53)

Why TLS SNI is visible even in encrypted traffic:
  The TLS handshake starts with a ClientHello message that is sent
  in PLAINTEXT (before encryption is established). This message
  includes the destination hostname so the server knows which
  certificate to present. This is called Server Name Indication (SNI).

TLS ClientHello structure (simplified):
  Byte 0:      Content Type  = 0x16 (Handshake)
  Bytes 1-2:   TLS Version   (0x0301 = TLS 1.0 record layer)
  Bytes 3-4:   Record Length
  Byte 5:      Handshake Type = 0x01 (Client Hello)
  Bytes 6-8:   Handshake Length
  Bytes 9-10:  Client Hello Version
  Bytes 11-42: Random (32 bytes)
  Byte 43:     Session ID Length (N)
  Bytes 44+N:  Cipher Suites Length, Cipher Suites
               Compression Methods Length, Methods
               Extensions Length
               [Extensions...]
               Extension Type 0x0000 = SNI
                 SNI List Length
                 SNI Entry Type = 0x00 (hostname)
                 SNI Length
                 SNI Value = "www.example.com"  ← WE WANT THIS
"""

import struct
from typing import Optional
from dataclasses import dataclass


# ── Result Types ──────────────────────────────────────────────────────────────

@dataclass
class DPIResult:
    """Result of deep packet inspection on a payload."""
    sni:          Optional[str] = None   # TLS Server Name Indication
    http_host:    Optional[str] = None   # HTTP Host header value
    dns_query:    Optional[str] = None   # DNS question name
    tls_version:  Optional[str] = None   # Negotiated TLS version
    ja3_hash:     Optional[str] = None   # JA3 fingerprint (Phase 2)
    source:       str = "unknown"        # "tls", "http", "dns"

    @property
    def domain(self) -> Optional[str]:
        """Return the best available domain name."""
        return self.sni or self.http_host or self.dns_query

    def __bool__(self) -> bool:
        return self.domain is not None


# ── TLS SNI Extractor ────────────────────────────────────────────────────────

class TLSSNIExtractor:
    """
    Extracts the Server Name Indication from a TLS ClientHello.

    Works by navigating the raw TLS record byte-by-byte.
    No external libraries required.
    """

    # TLS content types
    CONTENT_HANDSHAKE     = 0x16
    HANDSHAKE_CLIENT_HELLO = 0x01

    # TLS extension types
    EXT_SNI               = 0x0000
    EXT_SUPPORTED_VERSIONS = 0x002B

    # TLS version map
    TLS_VERSIONS = {
        0x0301: "TLS 1.0",
        0x0302: "TLS 1.1",
        0x0303: "TLS 1.2",
        0x0304: "TLS 1.3",
        0x0300: "SSL 3.0",
    }

    @classmethod
    def extract(cls, payload: bytes) -> Optional[DPIResult]:
        """
        Parse a TLS ClientHello and extract SNI + version info.
        Returns None if payload is not a valid ClientHello.
        """
        if len(payload) < 6:
            return None

        # ── Check TLS record header (5 bytes) ─────────────────────────────────
        content_type = payload[0]
        if content_type != cls.CONTENT_HANDSHAKE:
            return None

        # TLS record layer version (not the negotiated version)
        record_version = struct.unpack("!H", payload[1:3])[0]
        record_len     = struct.unpack("!H", payload[3:5])[0]

        if len(payload) < 5 + record_len:
            return None     # Truncated

        # ── Handshake layer ───────────────────────────────────────────────────
        handshake_type = payload[5]
        if handshake_type != cls.HANDSHAKE_CLIENT_HELLO:
            return None

        # Handshake length is 3 bytes (big-endian)
        if len(payload) < 9:
            return None
        hs_len = struct.unpack("!I", b'\x00' + payload[6:9])[0]
        offset = 9      # Start of ClientHello body

        # ── ClientHello body ──────────────────────────────────────────────────
        if offset + 2 > len(payload):
            return None
        client_version = struct.unpack("!H", payload[offset:offset+2])[0]
        offset += 2

        # Skip Random (32 bytes)
        offset += 32
        if offset >= len(payload):
            return None

        # Skip Session ID
        session_len = payload[offset]
        offset += 1 + session_len
        if offset + 2 > len(payload):
            return None

        # Skip Cipher Suites
        cipher_len = struct.unpack("!H", payload[offset:offset+2])[0]
        offset += 2 + cipher_len
        if offset >= len(payload):
            return None

        # Skip Compression Methods
        comp_len = payload[offset]
        offset += 1 + comp_len
        if offset + 2 > len(payload):
            return None

        # ── Extensions ────────────────────────────────────────────────────────
        ext_total_len = struct.unpack("!H", payload[offset:offset+2])[0]
        offset += 2
        ext_end = offset + ext_total_len

        sni_value    = None
        tls_ver_str  = cls.TLS_VERSIONS.get(client_version, f"0x{client_version:04x}")

        while offset + 4 <= ext_end and offset + 4 <= len(payload):
            ext_type = struct.unpack("!H", payload[offset:offset+2])[0]
            ext_len  = struct.unpack("!H", payload[offset+2:offset+4])[0]
            offset  += 4

            if offset + ext_len > len(payload):
                break

            ext_data = payload[offset : offset + ext_len]

            # ── SNI Extension (type 0x0000) ───────────────────────────────────
            if ext_type == cls.EXT_SNI and len(ext_data) >= 5:
                sni_value = cls._parse_sni_extension(ext_data)

            # ── Supported Versions Extension (type 0x002B) ────────────────────
            elif ext_type == cls.EXT_SUPPORTED_VERSIONS and len(ext_data) >= 3:
                # For TLS 1.3 the version is negotiated here
                inner_ver = cls._parse_supported_versions(ext_data)
                if inner_ver:
                    tls_ver_str = inner_ver

            offset += ext_len

        if sni_value is None:
            return None

        return DPIResult(
            sni=sni_value.lower().strip(),
            tls_version=tls_ver_str,
            source="tls",
        )

    @classmethod
    def _parse_sni_extension(cls, data: bytes) -> Optional[str]:
        """
        SNI extension wire format:
          2 bytes: ServerNameList length
          For each entry:
            1 byte:  name type (0x00 = hostname)
            2 bytes: name length
            N bytes: hostname ASCII
        """
        if len(data) < 5:
            return None
        try:
            list_len = struct.unpack("!H", data[0:2])[0]
            pos = 2
            while pos + 3 <= len(data) and pos + 3 <= 2 + list_len:
                name_type = data[pos]
                name_len  = struct.unpack("!H", data[pos+1:pos+3])[0]
                pos += 3
                if name_type == 0x00:    # hostname
                    return data[pos:pos+name_len].decode("ascii", errors="replace")
                pos += name_len
        except (struct.error, UnicodeDecodeError):
            pass
        return None

    @classmethod
    def _parse_supported_versions(cls, data: bytes) -> Optional[str]:
        """Parse TLS supported_versions extension (client side has list)."""
        try:
            if data[0] % 2 != 0:
                return None
            count = data[0] // 2
            for i in range(count):
                v = struct.unpack("!H", data[1+i*2 : 3+i*2])[0]
                if v in cls.TLS_VERSIONS:
                    return cls.TLS_VERSIONS[v]
        except (struct.error, IndexError):
            pass
        return None


# ── HTTP Host Extractor ───────────────────────────────────────────────────────

class HTTPHostExtractor:
    """
    Extracts the Host header from plaintext HTTP/1.x requests.

    Handles:
      GET /path HTTP/1.1\r\nHost: example.com\r\n...
      POST /api HTTP/1.0\r\nHost: api.site.com:8080\r\n...
    """

    HTTP_METHODS = (
        b"GET ", b"POST ", b"PUT ", b"DELETE ",
        b"HEAD ", b"OPTIONS ", b"PATCH ", b"CONNECT ",
    )

    @classmethod
    def extract(cls, payload: bytes) -> Optional[DPIResult]:
        """
        Returns DPIResult with http_host set, or None if not HTTP.
        """
        if not payload:
            return None

        # Quick check: must start with an HTTP method
        if not any(payload.startswith(m) for m in cls.HTTP_METHODS):
            return None

        # Find Host header (case-insensitive search)
        lower = payload.lower()
        host_idx = lower.find(b"\r\nhost:")
        if host_idx == -1:
            host_idx = lower.find(b"\nhost:")

        if host_idx == -1:
            return None

        # Skip past "host:" prefix
        colon_pos = payload.find(b":", host_idx + 2)
        if colon_pos == -1:
            return None

        end = payload.find(b"\r\n", colon_pos)
        if end == -1:
            end = payload.find(b"\n", colon_pos)
        if end == -1:
            end = len(payload)

        host_raw = payload[colon_pos+1:end].strip()
        try:
            host_str = host_raw.decode("ascii", errors="replace").strip()
            # Strip port if present (e.g. "example.com:8080" → "example.com")
            if ":" in host_str:
                host_str = host_str.split(":")[0]
            if not host_str:
                return None
            return DPIResult(http_host=host_str.lower(), source="http")
        except Exception:
            return None


# ── DNS Query Extractor ───────────────────────────────────────────────────────

class DNSExtractor:
    """
    Extracts the queried domain name from a DNS request packet.

    DNS message format:
      2 bytes: Transaction ID
      2 bytes: Flags
      2 bytes: Questions count
      2 bytes: Answer RRs
      2 bytes: Authority RRs
      2 bytes: Additional RRs
      [Questions section]
        QNAME: sequence of labels, each prefixed by length byte, ended by 0x00
        2 bytes: QTYPE
        2 bytes: QCLASS
    """

    @classmethod
    def extract(cls, payload: bytes) -> Optional[DPIResult]:
        """
        Parse DNS payload and return the queried domain name.
        Returns None if payload is not a valid DNS query.
        """
        if len(payload) < 12:
            return None

        try:
            flags     = struct.unpack("!H", payload[2:4])[0]
            questions = struct.unpack("!H", payload[4:6])[0]

            # QR bit: 0 = query, 1 = response
            is_query = not (flags & 0x8000)
            if not is_query or questions == 0:
                return None     # Only parse queries

            # Parse first question QNAME starting at offset 12
            name = cls._decode_labels(payload, 12)
            if name:
                return DPIResult(dns_query=name.lower(), source="dns")

        except (struct.error, IndexError):
            pass
        return None

    @classmethod
    def _decode_labels(cls, data: bytes, offset: int) -> Optional[str]:
        """
        DNS label encoding:
          Each label = 1-byte length + N bytes of ASCII
          0x00 = end of name
          0xC0 = pointer to another offset (compression)
        """
        labels = []
        visited = set()
        max_iter = 128  # Prevent infinite loops on malformed data

        while offset < len(data) and max_iter > 0:
            max_iter -= 1
            length = data[offset]

            if length == 0:
                break   # End of name

            # DNS compression pointer (0xC0 prefix)
            if (length & 0xC0) == 0xC0:
                if offset + 1 >= len(data):
                    return None
                pointer = struct.unpack("!H", data[offset:offset+2])[0] & 0x3FFF
                if pointer in visited:
                    return None  # Circular reference
                visited.add(pointer)
                offset = pointer
                continue

            offset += 1
            if offset + length > len(data):
                return None

            label = data[offset:offset+length]
            try:
                labels.append(label.decode("ascii"))
            except UnicodeDecodeError:
                labels.append(label.decode("latin-1"))
            offset += length

        return ".".join(labels) if labels else None


# ── Unified DPI Inspector ────────────────────────────────────────────────────

class DPIInspector:
    """
    Top-level inspector: tries all extractors on a payload.

    Priority: TLS SNI > HTTP Host > DNS Query

    Usage:
        result = DPIInspector.inspect(parsed_packet)
        if result:
            print(f"Domain: {result.domain}  Source: {result.source}")
    """

    @staticmethod
    def inspect(payload: bytes, src_port: int = 0, dst_port: int = 0) -> Optional[DPIResult]:
        """
        Try all DPI methods and return the first successful result.
        Port hints can guide which parser to try first.
        """
        if not payload:
            return None

        # ── DNS (port 53) ──────────────────────────────────────────────────────
        if src_port == 53 or dst_port == 53:
            result = DNSExtractor.extract(payload)
            if result:
                return result

        # ── HTTPS / TLS (port 443, 8443, 853, etc.) ───────────────────────────
        if dst_port in (443, 8443, 853) or src_port in (443, 8443, 853):
            result = TLSSNIExtractor.extract(payload)
            if result:
                return result

        # ── HTTP (port 80, 8080, etc.) ─────────────────────────────────────────
        if dst_port in (80, 8080, 8000) or src_port in (80, 8080, 8000):
            result = HTTPHostExtractor.extract(payload)
            if result:
                return result

        # ── Fallback: try all parsers regardless of port ───────────────────────
        result = TLSSNIExtractor.extract(payload)
        if result:
            return result

        result = HTTPHostExtractor.extract(payload)
        if result:
            return result

        result = DNSExtractor.extract(payload)
        if result:
            return result

        return None