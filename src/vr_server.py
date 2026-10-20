"""Quest browser <-> PC link for the VR teleop: one port serves the WebXR page and a WebSocket.

Standard library only (``http.server`` + ``ssl`` + a minimal RFC 6455 WebSocket), so nothing
has to be installed. The page (``vr_page.html``, next to this file) runs in the Quest browser,
reads both controllers every XR frame and sends them here; ``vr_teleop.py``
reads the newest sample with :meth:`VRLink.latest` and sends a status (the HUD text) and
haptic pulses back.

WebXR only runs on a secure page, hence HTTPS with a self-signed certificate, made once in
``revobots_arm/certs/`` (with the ``cryptography`` package if it is installed, else with the
``openssl`` command, which Git for Windows ships). The headset warns about it the first time:
Advanced -> Proceed. Plain HTTP (``https=False``) is for a USB cable with
``adb reverse tcp:8443 tcp:8443`` and ``http://localhost:8443`` in the headset: localhost
counts as secure.
"""

from __future__ import annotations

import base64
import hashlib
import ipaddress
import json
import logging
import os
import select
import shutil
import socket
import socketserver
import ssl
import struct
import subprocess
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from typing import Any

from ._bootstrap import ARM_ROOT

logger = logging.getLogger(__name__)

PAGE = Path(__file__).resolve().parent / "vr_page.html"
CERT_DIR = ARM_ROOT / "certs"
_WS_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"
_MAX_FRAME = 1 << 20
_STATUS_PERIOD_S = 0.2  # HUD updates at most this often ...
_STATUS_KEEPALIVE_S = 1.0  # ... and at least this often


# ── addresses + certificate ──────────────────────────────────────────────────────


def local_ipv4s() -> list[str]:
    """This PC's LAN addresses, the one on the default route first."""
    found: list[str] = []
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))  # sends nothing; only picks the outgoing interface
            found.append(s.getsockname()[0])
    except OSError:
        pass
    try:
        for ip in socket.gethostbyname_ex(socket.gethostname())[2]:
            if ip not in found:
                found.append(ip)
    except OSError:
        pass
    return [ip for ip in found if not ip.startswith(("127.", "169.254."))]


def ensure_certificate(cert_file: str | None = None, key_file: str | None = None) -> tuple[Path, Path]:
    """The given certificate + key, or the self-signed pair in ``certs/`` (made on first use)."""
    if cert_file or key_file:
        if not (cert_file and key_file):
            raise ValueError("teleop.cert_file and teleop.key_file go together")
        cert, key = Path(cert_file).expanduser(), Path(key_file).expanduser()
        for path in (cert, key):
            if not path.is_file():
                raise FileNotFoundError(f"vr_ee: {path} not found")
        return cert, key

    cert, key = CERT_DIR / "vr_cert.pem", CERT_DIR / "vr_key.pem"
    if cert.is_file() and key.is_file():
        return cert, key
    CERT_DIR.mkdir(parents=True, exist_ok=True)
    (CERT_DIR / ".gitignore").write_text("*\n")  # a private key: never commit it
    ips = local_ipv4s()
    try:
        _make_cert_cryptography(cert, key, ips)
    except ImportError:
        _make_cert_openssl(cert, key, ips)
    logger.info("vr_ee: made a self-signed certificate for the headset page: %s", cert)
    return cert, key


def _make_cert_cryptography(cert: Path, key: Path, ips: list[str]) -> None:
    import datetime

    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID

    pkey = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "revobots-arm-vr")])
    now = datetime.datetime.now(datetime.timezone.utc)
    san = [x509.DNSName("localhost")] + [x509.IPAddress(ipaddress.ip_address(ip)) for ip in ips]
    crt = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(pkey.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(days=1))
        .not_valid_after(now + datetime.timedelta(days=3650))
        .add_extension(x509.SubjectAlternativeName(san), critical=False)
        .sign(pkey, hashes.SHA256())
    )
    key.write_bytes(
        pkey.private_bytes(
            serialization.Encoding.PEM, serialization.PrivateFormat.TraditionalOpenSSL, serialization.NoEncryption()
        )
    )
    cert.write_bytes(crt.public_bytes(serialization.Encoding.PEM))


def _find_openssl() -> str | None:
    found = shutil.which("openssl")
    if found:
        return found
    roots = (os.environ.get("ProgramFiles"), os.environ.get("ProgramFiles(x86)"), os.environ.get("LOCALAPPDATA"))
    for root in filter(None, roots):
        for sub in (r"Git\usr\bin\openssl.exe", r"Git\mingw64\bin\openssl.exe", r"Programs\Git\usr\bin\openssl.exe"):
            path = Path(root) / sub
            if path.is_file():
                return str(path)
    return None


def _make_cert_openssl(cert: Path, key: Path, ips: list[str]) -> None:
    exe = _find_openssl()
    if exe is None:
        raise RuntimeError(
            "vr_ee: the headset page needs HTTPS (WebXR only runs on secure pages), and neither the "
            "`cryptography` package nor the `openssl` command was found to make a certificate. "
            "`pip install cryptography`, or install Git for Windows (it ships openssl), or pass your own "
            "--teleop.cert_file / --teleop.key_file, or use --teleop.https=false over USB (adb reverse)."
        )
    san = ",".join(["DNS:localhost"] + [f"IP:{ip}" for ip in ips])
    conf = (
        "[req]\nprompt = no\ndistinguished_name = dn\nx509_extensions = ext\n"
        "[dn]\nCN = revobots-arm-vr\n"
        f"[ext]\nsubjectAltName = {san}\nbasicConstraints = CA:FALSE\n"
    )
    with tempfile.TemporaryDirectory() as tmp:
        conf_path = Path(tmp) / "openssl.cnf"
        conf_path.write_text(conf)
        try:
            subprocess.run(
                [exe, "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-sha256", "-days", "3650",
                 "-keyout", str(key), "-out", str(cert), "-config", str(conf_path)],
                check=True, capture_output=True, timeout=60,
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"vr_ee: openssl could not make a certificate: {e.stderr.decode(errors='replace')}") from e


# ── minimal WebSocket (RFC 6455, server side) ────────────────────────────────────


def _recv_exact(sock, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("socket closed")
        buf += chunk
    return bytes(buf)


def _ws_read(sock) -> tuple[int, bytes]:
    """One complete message as ``(opcode, payload)``; control frames come back as they are."""
    message = bytearray()
    opcode = 0x1
    while True:
        b1, b2 = _recv_exact(sock, 2)
        op = b1 & 0x0F
        n = b2 & 0x7F
        if n == 126:
            n = struct.unpack(">H", _recv_exact(sock, 2))[0]
        elif n == 127:
            n = struct.unpack(">Q", _recv_exact(sock, 8))[0]
        if n > _MAX_FRAME:
            raise ValueError(f"websocket frame of {n} bytes")
        mask = _recv_exact(sock, 4) if b2 & 0x80 else None
        data = _recv_exact(sock, n) if n else b""
        if mask and n:
            pad = (mask * (n // 4 + 1))[:n]
            data = (int.from_bytes(data, "big") ^ int.from_bytes(pad, "big")).to_bytes(n, "big")
        if op >= 0x8:  # close / ping / pong: never fragmented
            return op, data
        if op != 0x0:
            opcode = op
        message += data
        if b1 & 0x80:
            return opcode, bytes(message)


def _ws_send(sock, payload: bytes, opcode: int = 0x1) -> None:
    n = len(payload)
    if n < 126:
        head = struct.pack(">BB", 0x80 | opcode, n)
    elif n < 1 << 16:
        head = struct.pack(">BBH", 0x80 | opcode, 126, n)
    else:
        head = struct.pack(">BBQ", 0x80 | opcode, 127, n)
    sock.sendall(head + payload)


def _readable(sock, timeout: float) -> bool:
    if isinstance(sock, ssl.SSLSocket) and sock.pending():
        return True
    readable, _, _ = select.select([sock], [], [], timeout)
    return bool(readable)


# ── HTTP server ──────────────────────────────────────────────────────────────────


class _Server(socketserver.ThreadingTCPServer):
    daemon_threads = True
    block_on_close = False
    allow_reuse_address = os.name != "nt"  # on Windows it would let a second teleop bind the same port

    def __init__(self, addr: tuple[str, int], link: VRLink, ssl_context: ssl.SSLContext | None) -> None:
        self.link = link
        self.ssl_context = ssl_context
        super().__init__(addr, _Handler)

    def get_request(self):
        sock, addr = super().get_request()
        if self.ssl_context is not None:  # the TLS handshake runs in the handler thread, not here
            sock = self.ssl_context.wrap_socket(sock, server_side=True, do_handshake_on_connect=False)
        return sock, addr

    def handle_error(self, request, client_address) -> None:
        # e.g. the browser refusing the self-signed certificate before you tap "Proceed"
        logger.debug("vr_ee: connection from %s failed", client_address, exc_info=True)


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    disable_nagle_algorithm = True
    timeout = 10.0
    server: _Server

    def setup(self) -> None:
        if isinstance(self.request, ssl.SSLSocket):
            self.request.settimeout(self.timeout)
            self.request.do_handshake()
        super().setup()

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        logger.debug("vr_ee http: " + format, *args)

    def do_GET(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0]
        if path == "/ws":
            self._websocket()
        elif path in ("/", "/index.html"):
            self._page()
        else:
            self.send_error(404)

    def _page(self) -> None:
        try:
            body = PAGE.read_bytes()
        except OSError:
            self.send_error(500, "vr_page.html is missing next to vr_server.py")
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _websocket(self) -> None:
        key = self.headers.get("Sec-WebSocket-Key", "")
        if not key or "websocket" not in self.headers.get("Upgrade", "").lower():
            self.send_error(400, "websocket upgrade expected")
            return
        accept = base64.b64encode(hashlib.sha1((key + _WS_GUID).encode()).digest()).decode()
        self.send_response(101, "Switching Protocols")
        self.send_header("Upgrade", "websocket")
        self.send_header("Connection", "Upgrade")
        self.send_header("Sec-WebSocket-Accept", accept)
        self.end_headers()
        self.close_connection = True
        self.server.link._serve_client(self.connection, self.client_address[0])


# ── the link ─────────────────────────────────────────────────────────────────────


class VRLink:
    """Serves the page, keeps the newest headset message, sends status and haptics back.

    One headset at a time: a newly opened page takes over from the previous one.
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8443,
        https: bool = True,
        cert_file: str | None = None,
        key_file: str | None = None,
        hand: str = "right",
    ) -> None:
        self.host = host
        self.port = int(port)
        self.https = https
        self.cert_file = cert_file
        self.key_file = key_file
        self.hand = hand
        self.urls: list[str] = []
        self._lock = threading.Lock()
        self._stopping = threading.Event()
        self._latest: tuple[dict[str, Any], float] | None = None
        self._client_gen = 0
        self._client: str | None = None
        self._status: dict[str, Any] | None = None
        self._status_dirty = False
        self._haptics: list[tuple[float, float]] = []
        self._server: _Server | None = None
        self._thread: threading.Thread | None = None

    # -- lifecycle ------------------------------------------------------------------

    def start(self) -> list[str]:
        ctx = None
        if self.https:
            cert, key = ensure_certificate(self.cert_file, self.key_file)
            ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            ctx.load_cert_chain(str(cert), str(key))
        try:
            self._server = _Server((self.host, self.port), self, ctx)
        except OSError as e:
            raise ConnectionError(
                f"vr_ee: cannot listen on {self.host}:{self.port} ({e}). Another teleop running? "
                "Try --teleop.port=8444."
            ) from e
        self._thread = threading.Thread(
            target=self._server.serve_forever, kwargs={"poll_interval": 0.2}, name="vr_ee-http", daemon=True
        )
        self._thread.start()
        scheme = "https" if self.https else "http"
        hosts = local_ipv4s() if self.host in ("", "0.0.0.0") else [self.host]
        if not self.https:
            hosts = ["localhost", *hosts]
        self.urls = [f"{scheme}://{h}:{self.port}" for h in hosts] or [f"{scheme}://<this-pc-ip>:{self.port}"]
        return self.urls

    def stop(self) -> None:
        self._stopping.set()
        with self._lock:
            self._client_gen += 1  # ends the client loop
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    # -- teleop side ----------------------------------------------------------------

    @property
    def connected(self) -> bool:
        with self._lock:
            return self._client is not None

    def latest(self, max_age_s: float) -> dict[str, Any] | None:
        """Newest message from the page, or None if there is none younger than ``max_age_s``."""
        with self._lock:
            item = self._latest
        if item is None or time.perf_counter() - item[1] > max_age_s:
            return None
        return item[0]

    def set_status(self, status: dict[str, Any]) -> None:
        with self._lock:
            if status != self._status:
                self._status = status
                self._status_dirty = True

    def haptic(self, intensity: float, duration_ms: float) -> None:
        with self._lock:
            if self._client is not None and len(self._haptics) < 8:
                self._haptics.append((float(intensity), float(duration_ms)))

    # -- client thread --------------------------------------------------------------

    def _serve_client(self, sock, addr: str) -> None:
        with self._lock:
            self._client_gen += 1
            gen = self._client_gen
            self._client = addr
            self._latest = None
            self._status_dirty = True
            self._haptics.clear()
        logger.info("vr_ee: headset page connected from %s", addr)
        last_status = -1.0
        try:
            sock.settimeout(2.0)
            _ws_send(sock, json.dumps({"hand": self.hand}).encode())
            while not self._stopping.is_set():
                out: list[str] = []
                with self._lock:
                    if gen != self._client_gen:
                        break  # a newer page took over
                    now = time.perf_counter()
                    out.extend(json.dumps({"haptic": list(h)}) for h in self._haptics)
                    self._haptics.clear()
                    due = now - last_status >= (_STATUS_PERIOD_S if self._status_dirty else _STATUS_KEEPALIVE_S)
                    if self._status is not None and due:
                        out.append(json.dumps({"status": self._status}))
                        self._status_dirty = False
                        last_status = now
                for text in out:
                    _ws_send(sock, text.encode())
                if not _readable(sock, 0.02):
                    continue
                opcode, payload = _ws_read(sock)
                if opcode == 0x8:  # close
                    try:
                        _ws_send(sock, payload[:2], opcode=0x8)
                    except OSError:
                        pass
                    break
                if opcode == 0x9:  # ping
                    _ws_send(sock, payload, opcode=0xA)
                    continue
                if opcode != 0x1:
                    continue
                try:
                    msg = json.loads(payload)
                except ValueError:
                    continue
                if isinstance(msg, dict) and msg.get("type") == "pose":
                    with self._lock:
                        if gen == self._client_gen:
                            self._latest = (msg, time.perf_counter())
        except (OSError, ValueError) as e:
            logger.debug("vr_ee: headset %s: %s", addr, e)
        finally:
            with self._lock:
                if gen == self._client_gen:
                    self._client = None
                    self._latest = None
            logger.info("vr_ee: headset page %s disconnected", addr)
