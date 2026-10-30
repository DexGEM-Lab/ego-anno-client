"""TLS transport, certificate pinning, multipart streaming, and retries."""
import hashlib
import http.client
import json
import os
from pathlib import Path
import ssl
import sys
import time
import uuid

API_HOST = "42.194.202.230"
API_PORT = 443
PINNED_SHA256 = "954C91ACBC570A3DDF2304FA153442D59270CBE10AD4763011E051400D08C2CD"
CHUNK_SIZE = 1024 * 1024
MAX_ATTEMPTS = 10


class ClientError(Exception):
    pass


class PinError(ClientError):
    pass


class RetryableError(ClientError):
    pass


def connect():
    """Authenticate the peer before http.client writes any request bytes."""
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    context.check_hostname = False
    context.verify_mode = ssl.CERT_NONE
    host = os.environ.get("EGO_ANNO_API_HOST", API_HOST)
    pin = os.environ.get("EGO_ANNO_PIN_SHA256", PINNED_SHA256)
    conn = http.client.HTTPSConnection(host, API_PORT, context=context, timeout=120)
    try:
        conn.connect()
        actual = hashlib.sha256(conn.sock.getpeercert(binary_form=True)).hexdigest().upper()
        if actual != pin:
            raise PinError("server certificate does not match the pinned fingerprint")
        return conn
    except BaseException:
        conn.close()
        raise


def headers(conn, method, path, key, extra=None):
    conn.putrequest(method, path)
    if key is not None:
        conn.putheader("Authorization", "Bearer " + key)
    for name, value in (extra or {}).items():
        conn.putheader(name, str(value))
    conn.endheaders()


def error_message(response):
    body = response.read(1024 * 1024)
    try:
        error = json.loads(body).get("error", {})
        if not isinstance(error, dict):
            error = {}
        detail = "{}: {}".format(error.get("code", "unknown_error"), error.get("message", ""))
    except (ValueError, AttributeError):
        detail = "unparseable error response"
    return "HTTP {}: {}".format(response.status, detail)


def with_retries(operation):
    for attempt in range(MAX_ATTEMPTS):
        try:
            return operation()
        except RetryableError as exc:
            message = str(exc)
        except (OSError, http.client.HTTPException) as exc:
            message = "network error: " + str(exc)
        if attempt == MAX_ATTEMPTS - 1:
            raise ClientError("{} (retry limit reached)".format(message))
        delay = min(2 ** attempt, 60)
        print("{}; retrying in {}s".format(message, delay), file=sys.stderr)
        time.sleep(delay)


def check_response(response, expected):
    if response.status == expected:
        return
    message = error_message(response)
    if response.status == 429 or 500 <= response.status <= 599:
        raise RetryableError(message)
    raise ClientError(message)


def filename_header(path):
    # Keep the basename inert and prevent multipart header injection.
    name = Path(path).name
    if not name or any(c in name for c in '\r\n"\\'):
        raise ClientError("unsupported characters in filename")
    return name


class Multipart:
    def __init__(self, fields, files):
        self.boundary = "ego-anno-" + uuid.uuid4().hex
        self.parts = []
        self.open_files = []
        self.length = 0
        try:
            for name, value in fields:
                data = ('--{}\r\nContent-Disposition: form-data; name="{}"\r\n\r\n{}\r\n'.format(
                    self.boundary, name, value)).encode("utf-8")
                self.parts.append(data)
                self.length += len(data)
            for name, path, content_type in files:
                filename = filename_header(path)
                handle = open(path, "rb")
                self.open_files.append(handle)
                stat = os.fstat(handle.fileno())
                head = ('--{}\r\nContent-Disposition: form-data; name="{}"; filename="{}"\r\n'
                        'Content-Type: {}\r\n\r\n'.format(self.boundary, name, filename, content_type)).encode("utf-8")
                self.parts.extend([head, (handle, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns), b"\r\n"])
                self.length += len(head) + stat.st_size + 2
            tail = ('--{}--\r\n'.format(self.boundary)).encode("ascii")
            self.parts.append(tail)
            self.length += len(tail)
        except BaseException:
            self.close()
            raise

    def close(self):
        for handle in self.open_files:
            handle.close()

    def send(self, conn):
        sent = 0
        last_percent = -5
        last_time = time.monotonic()
        for part in self.parts:
            if isinstance(part, bytes):
                conn.send(part)
                sent += len(part)
                continue
            handle, size, mtime, ctime = part
            stat = os.fstat(handle.fileno())
            if (stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns) != (size, mtime, ctime):
                raise ClientError("input file changed during submission; use a new job and idempotency key")
            handle.seek(0)
            remaining = size
            while remaining:
                chunk = handle.read(min(CHUNK_SIZE, remaining))
                if not chunk:
                    raise ClientError("input file changed during submission")
                conn.send(chunk)
                remaining -= len(chunk)
                sent += len(chunk)
                now = time.monotonic()
                percent = sent * 100 // self.length
                if percent >= last_percent + 5 or now - last_time >= 10:
                    print("Upload: {}%".format(percent), file=sys.stderr)
                    last_percent, last_time = percent, now
            stat = os.fstat(handle.fileno())
            if (stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns) != (size, mtime, ctime):
                raise ClientError("input file changed during submission; use a new job and idempotency key")
