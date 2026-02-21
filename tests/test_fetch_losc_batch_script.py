from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path


SCRIPT = Path("tools/fetch_losc_batch.sh").resolve()


def _write_mock_curl(path: Path) -> None:
    path.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
args=("$@")
url="${args[$(($#-1))]}"

# HEAD/status checks used by resolve_short_name()
if [[ "$*" == *"%{http_code}"* ]]; then
  if [[ "$url" == *"GW150914-v2" ]]; then
    printf '200'
  elif [[ "$url" == *"GW151226-v2" ]]; then
    printf '404'
  elif [[ "$url" == *"GW151226-v1" ]]; then
    printf '200'
  else
    printf '404'
  fi
  exit 0
fi

if [[ "$url" == *"GW150914/v2" ]]; then
  cat <<'JSON'
{"events":{"GW150914-v2":{"strain":[{"url":"https://example/H-H1_GW150914.hdf5"},{"url":"https://example/L-L1_GW150914.hdf5"}]}}}
JSON
  exit 0
fi

if [[ "$url" == *"GW151226/v1" ]]; then
  cat <<'JSON'
{"events":{"GW151226-v1":{"strain":[{"url":"https://example/H-H1_GW151226.hdf5"},{"url":"https://example/L-L1_GW151226.hdf5"}]}}}
JSON
  exit 0
fi

# file downloads
if [[ "$url" == https://example/* ]]; then
  out=''
  for ((i=1; i <= $#; i++)); do
    if [[ "${!i}" == "-o" ]]; then
      j=$((i+1))
      out="${!j}"
      break
    fi
  done
  : "${out:?missing -o output path}"
  printf 'dummy data' > "$out"
  exit 0
fi

echo "unexpected curl invocation: $*" >&2
exit 1
""",
        encoding="utf-8",
    )
    path.chmod(path.stat().st_mode | stat.S_IEXEC)


def test_fetch_losc_batch_resolves_v2_then_v1_and_is_idempotent(tmp_path: Path) -> None:
    mock_curl = tmp_path / "mock_curl.sh"
    _write_mock_curl(mock_curl)

    env = os.environ.copy()
    env["CURL_BIN"] = str(mock_curl)
    env["LOSC_ROOT"] = str(tmp_path / "data" / "losc")

    first = subprocess.run(
        [str(SCRIPT), "GW150914", "GW151226"],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert first.returncode == 0, first.stderr
    assert "shortName=GW150914/v2" in first.stdout
    assert "shortName=GW151226/v1" in first.stdout

    assert (tmp_path / "data" / "losc" / "GW150914" / "H-H1_GW150914.hdf5").exists()
    assert (tmp_path / "data" / "losc" / "GW150914" / "L-L1_GW150914.hdf5").exists()
    assert (tmp_path / "data" / "losc" / "GW151226" / "H-H1_GW151226.hdf5").exists()
    assert (tmp_path / "data" / "losc" / "GW151226" / "L-L1_GW151226.hdf5").exists()

    second = subprocess.run(
        [str(SCRIPT), "GW150914", "GW151226"],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert second.returncode == 0
    assert "SKIP GW150914" in second.stdout
    assert "SKIP GW151226" in second.stdout
