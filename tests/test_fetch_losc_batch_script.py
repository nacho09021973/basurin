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

if [[ "$*" == *"%{http_code}"* ]]; then
  if [[ "$url" == *"GW150914-v2" ]]; then
    printf '200'
  elif [[ "$url" == *"GW151226-v2" ]]; then
    printf '404'
  elif [[ "$url" == *"GW151226-v1" ]]; then
    printf '200'
  elif [[ "$url" == *"GW170729-v2" ]]; then
    printf '200'
  elif [[ "$url" == *"GW170823-v2" ]]; then
    printf '200'
  elif [[ "$url" == *"GW170823-v1" ]]; then
    printf '200'
  elif [[ "$url" == *"GW170809-v2" ]]; then
    printf '404'
  elif [[ "$url" == *"GW170809-v1" ]]; then
    printf '404'
  else
    printf '404'
  fi
  exit 0
fi

if [[ "$url" == *"GW150914-v2/strain-files?detector=H1" ]]; then
  cat <<'JSON'
{"results":[{"download_url":"https://example/H-H1_bad_rate.hdf5","duration":32,"sample_rate":2048},{"download_url":"https://example/H-H1_pref.hdf5","duration":32,"sample_rate":4096}]}
JSON
  exit 0
fi
if [[ "$url" == *"GW150914-v2/strain-files?detector=L1" ]]; then
  cat <<'JSON'
{"results":[{"download_url":"https://example/L-L1_ok.hdf5"}]}
JSON
  exit 0
fi
if [[ "$url" == *"GW150914-v2/strain-files?detector=V1" ]]; then
  echo "<html>not json</html>"
  exit 0
fi

if [[ "$url" == *"GW151226-v1/strain-files?detector=H1" ]]; then
  cat <<'JSON'
{"strain_files":[{"download_url":"https://example/H-H1_151226.hdf5","duration":16,"sample_rate":4096}]}
JSON
  exit 0
fi
if [[ "$url" == *"GW151226-v1/strain-files?detector=L1" ]]; then
  echo ""
  exit 0
fi
if [[ "$url" == *"GW151226-v1/strain-files?detector=V1" ]]; then
  cat <<'JSON'
{"results":[]}
JSON
  exit 0
fi

if [[ "$url" == *"GW170729-v2/strain-files?detector=H1" ]]; then
  cat <<'JSON'
{"results_count":1,"results":[{"download_url":"https://example/H-H1_170729.hdf5","file_format":"HDF5"}]}
JSON
  exit 0
fi
if [[ "$url" == *"GW170729-v2/strain-files?detector=L1" ]]; then
  cat <<'JSON'
{"results":[{"download_url":"https://example/L-L1_170729.hdf5"}]}
JSON
  exit 0
fi
if [[ "$url" == *"GW170729-v2/strain-files?detector=V1" ]]; then
  cat <<'JSON'
{"results":[{"download_url":"https://example/V-V1_170729.txt"}]}
JSON
  exit 0
fi

if [[ "$url" == *"GW170823-v2/strain-files?detector=H1" ]]; then
  cat <<'JSON'
{"results_count":0,"results":[]}
JSON
  exit 0
fi
if [[ "$url" == *"GW170823-v1/strain-files?detector=H1" ]]; then
  cat <<'JSON'
{"results_count":2,"results":[{"download_url":"https://example/H-H1_170823.gwf","file_format":"GWF"},{"download_url":"https://example/H-H1_170823.hdf5","file_format":"HDF5"}]}
JSON
  exit 0
fi
if [[ "$url" == *"GW170823-v1/strain-files?detector=L1" ]]; then
  cat <<'JSON'
{"results":[{"download_url":"https://example/L-L1_170823.gwf","file_format":"GWF"}]}
JSON
  exit 0
fi
if [[ "$url" == *"GW170823-v1/strain-files?detector=V1" ]]; then
  echo "not-json"
  exit 0
fi

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


def test_fetch_losc_batch_file_input_and_idempotent(tmp_path: Path) -> None:
    mock_curl = tmp_path / "mock_curl.sh"
    _write_mock_curl(mock_curl)

    events_file = tmp_path / "events.txt"
    events_file.write_text(
        """
# sample events
GW150914

GW151226
GW170729
GW170823
GW170809
""",
        encoding="utf-8",
    )

    env = os.environ.copy()
    env["CURL_BIN"] = str(mock_curl)
    env["LOSC_ROOT"] = str(tmp_path / "data" / "losc")

    first = subprocess.run(
        [str(SCRIPT), str(events_file)],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert first.returncode == 0, first.stderr
    assert "shortName=GW150914-v2" in first.stdout
    assert "shortName=GW151226-v1" in first.stdout
    assert "shortName=GW170823-v1" in first.stdout
    assert "GW170809" in first.stderr

    assert (tmp_path / "data" / "losc" / "GW150914" / "H-H1_pref.hdf5").exists()
    assert (tmp_path / "data" / "losc" / "GW150914" / "L-L1_ok.hdf5").exists()
    assert not (tmp_path / "data" / "losc" / "GW150914" / "H-H1_bad_rate.hdf5").exists()
    assert (tmp_path / "data" / "losc" / "GW151226" / "H-H1_151226.hdf5").exists()
    assert (tmp_path / "data" / "losc" / "GW170729" / "L-L1_170729.hdf5").exists()
    assert (tmp_path / "data" / "losc" / "GW170823" / "H-H1_170823.hdf5").exists()
    assert not (tmp_path / "data" / "losc" / "GW170823" / "H-H1_170823.gwf").exists()

    sha_file = tmp_path / "data" / "losc" / "GW150914" / "SHA256SUMS.txt"
    assert sha_file.exists()
    assert "H-H1_pref.hdf5" in sha_file.read_text(encoding="utf-8")

    second = subprocess.run(
        [str(SCRIPT), str(events_file)],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert second.returncode == 0
    assert "GW150914 H1: SKIP H-H1_pref.hdf5" in second.stdout
    assert "GW151226 H1: SKIP H-H1_151226.hdf5" in second.stdout
    assert "GW170823 H1: SKIP H-H1_170823.hdf5" in second.stdout
