"""Small HTTPS download utility: verified TLS and atomic local files."""
from __future__ import annotations

import os
from pathlib import Path
import ssl
import subprocess
import urllib.error
import urllib.request


def download_https(url: str, destination: Path, *, timeout: int = 120) -> None:
    if not url.startswith("https://"):
        raise ValueError("Weather downloads require HTTPS.")
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix+".partial")
    try:
        try:
            with urllib.request.urlopen(url, timeout=timeout) as response, partial.open("wb") as out:
                while block := response.read(1024*1024):
                    out.write(block)
        except urllib.error.URLError as error:
            if os.name != "nt" or not isinstance(error.reason, ssl.SSLCertVerificationError):
                raise
            quote = lambda value: "'"+str(value).replace("'", "''")+"'"
            script = ("$ErrorActionPreference='Stop'; Invoke-WebRequest -Uri "+quote(url)
                      +" -OutFile "+quote(partial.resolve())+f" -TimeoutSec {int(timeout)}")
            subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script], check=True)
        partial.replace(destination)
    finally:
        partial.unlink(missing_ok=True)
