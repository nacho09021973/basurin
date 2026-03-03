<#
.SYNOPSIS
    Download GWOSC strain HDF5 files for BASURIN ringdown analysis.

.DESCRIPTION
    Iterates over a list of LIGO BBH events, queries the GWOSC API v2,
    and downloads H1/L1 strain files (HDF format, duration=32s, 4KHz preferred).

    API chain per event:
      1) GET /api/v2/events/<EVENT_ID>           -> versions[] (iterate backwards)
      2) GET <version.detail_url>                -> strain_files_url
      3) GET <strain_files_url> (paginated)      -> list of download URLs
      4) Select best file per detector: HDF, duration=32, 4KHz preferred

    Known API quirks handled:
      - GWTC-2.1-confident versions often have 0 strain files -> fall back
      - Field is "file_format" with value "HDF" (not "format"/"hdf5")
      - strain_files_url returns paginated {next, results, results_count}
      - GW190521_030229 resolves to GW190521 via aliases

.PARAMETER StrictDuration32
    $true  -> skip detectors lacking duration=32 HDF files
    $false -> fall back to shortest available HDF duration (default)

.PARAMETER PreferredSampleRateKHz
    Preferred sample rate in kHz. Default: 4 (sufficient for ringdown at 200-400 Hz).
    Falls back to any available rate if preferred not found.

.EXAMPLE
    Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
    .\download_gwosc_strain.ps1

    # Strict mode + debug output:
    .\download_gwosc_strain.ps1 -StrictDuration32 $true -DebugCandidates

    # Self-check only (no downloads):
    .\download_gwosc_strain.ps1 -SelfCheckOnly

.NOTES
    Output: data\losc\<EVENT_ID>\{H1,L1}.hdf5
    UNC workaround: copy script to C:\tmp if running from \\wsl$\...
#>

param(
    [bool]$StrictDuration32 = $false,
    [int]$PreferredSampleRateKHz = 4,
    [int]$MaxRetries = 3,
    [int]$BackoffBaseSeconds = 2,
    [switch]$DebugCandidates,
    [switch]$SelfCheckOnly
)

# ============================================================================
# Configuration
# ============================================================================

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"   # speeds up Invoke-WebRequest

$EVENTS = @(
    "GW150914",
    "GW151226",
    "GW170104",
    "GW170608",
    "GW170729",
    "GW170809",
    "GW170814",
    "GW170818",
    "GW170823",
    "GW190521_030229"
)

$API_BASE = "https://gwosc.org/api/v2"
$DETECTORS = @("H1", "L1")
$PREFERRED_FORMAT = "HDF"          # API field value (not "hdf5")
$PREFERRED_DURATION = 32
$OUTPUT_ROOT = Join-Path (Get-Location) "data" "losc"

# ============================================================================
# Helpers
# ============================================================================

function Write-Log {
    param([string]$Tag, [string]$Message)
    $ts = Get-Date -Format "HH:mm:ss"
    Write-Host "[$ts] [$Tag] $Message"
}

function Invoke-ApiGet {
    <#
    .SYNOPSIS
        GET a JSON endpoint with retries and exponential backoff.
    #>
    param([string]$Url)

    for ($attempt = 1; $attempt -le $MaxRetries; $attempt++) {
        try {
            $resp = Invoke-WebRequest -Uri $Url -UseBasicParsing -TimeoutSec 30
            return ($resp.Content | ConvertFrom-Json)
        }
        catch {
            $err = $_.Exception.Message
            if ($attempt -lt $MaxRetries) {
                $wait = $BackoffBaseSeconds * [math]::Pow(2, $attempt - 1)
                Write-Log "RETRY" "Attempt ${attempt}/${MaxRetries} failed for $Url -- waiting ${wait}s"
                Start-Sleep -Seconds $wait
            }
            else {
                throw "API call failed after $MaxRetries attempts: $Url -- $err"
            }
        }
    }
}

function Get-AllStrainFiles {
    <#
    .SYNOPSIS
        Fetch all pages from a paginated strain_files_url endpoint.
    #>
    param([string]$Url)

    $allResults = @()
    $currentUrl = $Url

    while ($currentUrl) {
        $page = Invoke-ApiGet -Url $currentUrl
        if ($page.results) {
            $allResults += $page.results
        }
        $currentUrl = $page.next   # null when no more pages
    }

    return $allResults
}

function Find-BestVersionWithStrain {
    <#
    .SYNOPSIS
        Iterate versions backwards until finding one with strain files.
        Returns (strain_file_list, version_number, catalog_name) or $null.
    #>
    param([object[]]$Versions, [string]$EventId)

    # Iterate from last to first
    $indices = ($Versions.Count - 1)..0
    foreach ($i in $indices) {
        $ver = $Versions[$i]
        $detailUrl = $ver.detail_url
        if (-not $detailUrl) { continue }

        try {
            $versionData = Invoke-ApiGet -Url $detailUrl
            $strainUrl = $versionData.strain_files_url
            if (-not $strainUrl) { continue }

            $files = Get-AllStrainFiles -Url $strainUrl
            if ($files -and $files.Count -gt 0) {
                $vNum = $ver.version
                $catalog = $ver.catalog
                Write-Log "API" "${EventId}: using v${vNum} (${catalog}) with $($files.Count) strain files"
                return @{
                    Files = $files
                    Version = $vNum
                    Catalog = $catalog
                }
            }
            else {
                Write-Log "INFO" "${EventId}: v$($ver.version) ($($ver.catalog)) has 0 strain files, trying older..."
            }
        }
        catch {
            Write-Log "WARN" "${EventId}: v$($ver.version) detail fetch failed: $($_.Exception.Message)"
        }
    }

    return $null
}

function Download-File {
    <#
    .SYNOPSIS
        Download a file. Try BITS first, fall back to Invoke-WebRequest with retries.
    #>
    param(
        [string]$Url,
        [string]$Destination
    )

    # Skip if already downloaded and non-empty
    if (Test-Path $Destination) {
        $existing = Get-Item $Destination
        if ($existing.Length -gt 0) {
            $sizeMB = '{0:N1}' -f ($existing.Length / 1MB)
            Write-Log "SKIP" "Already exists (${sizeMB} MB): $(Split-Path $Destination -Leaf)"
            return $true
        }
        else {
            Remove-Item $Destination -Force
        }
    }

    # Ensure parent directory exists
    $parentDir = Split-Path $Destination -Parent
    if (-not (Test-Path $parentDir)) {
        New-Item -ItemType Directory -Path $parentDir -Force | Out-Null
    }

    # Try BITS transfer first (faster, resumable)
    $bitsOk = $false
    try {
        if (Get-Command Start-BitsTransfer -ErrorAction SilentlyContinue) {
            Write-Log "BITS" "Downloading -> $(Split-Path $Destination -Leaf)"
            Start-BitsTransfer -Source $Url -Destination $Destination -ErrorAction Stop
            if ((Test-Path $Destination) -and ((Get-Item $Destination).Length -gt 0)) {
                $bitsOk = $true
            }
        }
    }
    catch {
        Write-Log "WARN" "BITS failed: $($_.Exception.Message) -- falling back to WebRequest"
        if (Test-Path $Destination) { Remove-Item $Destination -Force -ErrorAction SilentlyContinue }
    }

    if ($bitsOk) { return $true }

    # Fallback: Invoke-WebRequest with retries
    for ($attempt = 1; $attempt -le $MaxRetries; $attempt++) {
        try {
            Write-Log "HTTP" "Download attempt ${attempt}/${MaxRetries} -> $(Split-Path $Destination -Leaf)"
            Invoke-WebRequest -Uri $Url -OutFile $Destination -UseBasicParsing -TimeoutSec 600
            if ((Test-Path $Destination) -and ((Get-Item $Destination).Length -gt 0)) {
                return $true
            }
            throw "Downloaded file is empty or missing"
        }
        catch {
            if (Test-Path $Destination) { Remove-Item $Destination -Force -ErrorAction SilentlyContinue }
            if ($attempt -lt $MaxRetries) {
                $wait = $BackoffBaseSeconds * [math]::Pow(2, $attempt - 1)
                Write-Log "RETRY" "Download failed -- waiting ${wait}s"
                Start-Sleep -Seconds $wait
            }
            else {
                Write-Log "ERROR" "Download failed after $MaxRetries attempts: $($_.Exception.Message)"
                return $false
            }
        }
    }
    return $false
}

function Select-StrainFile {
    <#
    .SYNOPSIS
        From a list of strain file entries, pick the best candidate for a detector.
        Preference: HDF format, duration=32, lowest sample rate >= PreferredSampleRateKHz.
        Returns $null if nothing suitable found.
    #>
    param(
        [object[]]$Files,
        [string]$Detector,
        [string]$EventId
    )

    # Filter: this detector + HDF format
    $hdfCandidates = @($Files | Where-Object {
        $_.detector -eq $Detector -and $_.file_format -eq $PREFERRED_FORMAT
    })

    if ($hdfCandidates.Count -eq 0) {
        if ($DebugCandidates) {
            Write-Log "DEBUG" "No HDF files for ${EventId}/${Detector}. All entries:"
            $allDet = @($Files | Where-Object { $_.detector -eq $Detector })
            foreach ($c in $allDet) {
                $bn = ($c.download_url -split '/')[-1]
                Write-Log "DEBUG" "  fmt=$($c.file_format) dur=$($c.duration) rate=$($c.sample_rate_kHz)kHz -> $bn"
            }
        }
        return $null
    }

    # Sub-filter: preferred duration
    $dur32 = @($hdfCandidates | Where-Object { $_.duration -eq $PREFERRED_DURATION })

    if ($dur32.Count -gt 0) {
        # Among duration=32, prefer the requested sample rate
        $preferred = @($dur32 | Where-Object { $_.sample_rate_kHz -eq $PreferredSampleRateKHz })
        if ($preferred.Count -gt 0) {
            return $preferred[0]
        }
        # Otherwise take the lowest sample rate available (smallest file)
        $sorted = @($dur32 | Sort-Object { [int]$_.sample_rate_kHz })
        return $sorted[0]
    }

    # No duration=32 HDF available
    if ($StrictDuration32) {
        $availDurations = ($hdfCandidates | ForEach-Object { $_.duration } | Sort-Object -Unique) -join ", "
        Write-Log "SKIP" "${EventId}/${Detector}: no HDF duration=32 (available durations: $availDurations)"
        if ($DebugCandidates) {
            foreach ($c in $hdfCandidates) {
                $bn = ($c.download_url -split '/')[-1]
                Write-Log "DEBUG" "  dur=$($c.duration) rate=$($c.sample_rate_kHz)kHz -> $bn"
            }
        }
        return $null
    }

    # Flexible: pick shortest available duration, then lowest sample rate
    $sorted = @($hdfCandidates | Sort-Object { [int]$_.duration }, { [int]$_.sample_rate_kHz })
    $chosen = $sorted[0]
    Write-Log "INFO" "${EventId}/${Detector}: no duration=32; using duration=$($chosen.duration) rate=$($chosen.sample_rate_kHz)kHz (fallback)"
    return $chosen
}

function Get-FileSha256 {
    param([string]$FilePath)
    if (Test-Path $FilePath) {
        return (Get-FileHash -Path $FilePath -Algorithm SHA256).Hash
    }
    return "FILE_NOT_FOUND"
}

# ============================================================================
# Self-check: inventory of files on disk
# ============================================================================

function Invoke-SelfCheck {
    Write-Host ""
    Write-Host "=========================================="
    Write-Host "  SELF-CHECK: strain files on disk"
    Write-Host "=========================================="
    Write-Host ""

    $totalOk = 0
    $totalMiss = 0

    foreach ($eventId in $EVENTS) {
        $eventDir = Join-Path $OUTPUT_ROOT $eventId
        $status = @{}

        foreach ($det in $DETECTORS) {
            $found = $false
            $foundInfo = ""

            if (Test-Path $eventDir) {
                # Look for canonical name first, then any matching file
                $canonical = Join-Path $eventDir "${det}.hdf5"
                if ((Test-Path $canonical) -and ((Get-Item $canonical).Length -gt 0)) {
                    $f = Get-Item $canonical
                    $found = $true
                    $sizeMB = '{0:N1}' -f ($f.Length / 1MB)
                    $foundInfo = "${det}.hdf5 (${sizeMB} MB)"
                }
                else {
                    $others = Get-ChildItem -Path $eventDir -File -ErrorAction SilentlyContinue |
                        Where-Object { $_.Name -match "$det" -and $_.Name -match "\.(hdf5|h5)$" -and $_.Length -gt 0 }
                    if ($others) {
                        $f = $others[0]
                        $found = $true
                        $sizeMB = '{0:N1}' -f ($f.Length / 1MB)
                        $foundInfo = "$($f.Name) (${sizeMB} MB)"
                    }
                }
            }

            if ($found) {
                $status[$det] = "[OK] $foundInfo"
                $totalOk++
            }
            else {
                $status[$det] = "[MISS]"
                $totalMiss++
            }
        }

        Write-Host "  ${eventId}:"
        Write-Host "    H1: $($status['H1'])"
        Write-Host "    L1: $($status['L1'])"
    }

    $total = $EVENTS.Count * $DETECTORS.Count
    Write-Host ""
    Write-Host "  Total: $totalOk OK, $totalMiss MISSING (of $total expected)"
    Write-Host ""
}

# ============================================================================
# Main download logic
# ============================================================================

function Invoke-DownloadAll {
    Write-Host ""
    Write-Host "=========================================="
    Write-Host "  GWOSC Strain Downloader for BASURIN"
    Write-Host "=========================================="
    Write-Host "  Output root     : $OUTPUT_ROOT"
    Write-Host "  Events          : $($EVENTS.Count)"
    Write-Host "  Strict dur=32   : $StrictDuration32"
    Write-Host "  Preferred rate  : ${PreferredSampleRateKHz} kHz"
    Write-Host "  Max retries     : $MaxRetries"
    Write-Host "=========================================="
    Write-Host ""

    # UNC path warning
    $cwd = (Get-Location).Path
    if ($cwd -match "^\\\\") {
        Write-Log "WARN" "Running from UNC path ($cwd)."
        Write-Log "WARN" "BITS transfer may fail on UNC. If downloads fail,"
        Write-Log "WARN" "copy this script to a local path (e.g. C:\tmp) and rerun."
    }

    $summary = @{}

    foreach ($eventId in $EVENTS) {
        Write-Host ""
        Write-Host ("=" * 55)
        Write-Log "EVENT" "Processing $eventId"
        Write-Host ("=" * 55)

        $eventDir = Join-Path $OUTPUT_ROOT $eventId
        if (-not (Test-Path $eventDir)) {
            New-Item -ItemType Directory -Path $eventDir -Force | Out-Null
        }

        $eventStatus = @{ "H1" = "PENDING"; "L1" = "PENDING" }

        try {
            # ---- Step 1: GET event metadata ----
            $eventUrl = "${API_BASE}/events/${eventId}"
            Write-Log "API" "GET $eventUrl"
            $eventData = Invoke-ApiGet -Url $eventUrl

            $versions = $eventData.versions
            if (-not $versions -or $versions.Count -eq 0) {
                throw "No versions found in API response"
            }
            Write-Log "INFO" "Found $($versions.Count) version(s): $(($versions | ForEach-Object { 'v' + $_.version + '(' + $_.catalog + ')' }) -join ', ')"

            # ---- Step 2-3: Find version with strain files (iterate backwards) ----
            $result = Find-BestVersionWithStrain -Versions $versions -EventId $eventId

            if (-not $result) {
                throw "No version has downloadable strain files"
            }

            $fileList = $result.Files

            # Debug: show all candidates
            if ($DebugCandidates) {
                Write-Log "DEBUG" "All strain file candidates for ${eventId} (v$($result.Version)):"
                foreach ($sf in $fileList) {
                    $bn = ($sf.download_url -split '/')[-1]
                    Write-Log "DEBUG" "  det=$($sf.detector) fmt=$($sf.file_format) dur=$($sf.duration) rate=$($sf.sample_rate_kHz)kHz -> $bn"
                }
            }

            # ---- Step 4: Select and download per detector ----
            foreach ($det in $DETECTORS) {
                $selected = Select-StrainFile -Files $fileList -Detector $det -EventId $eventId

                if (-not $selected) {
                    $eventStatus[$det] = "MISS"
                    Write-Log "MISS" "${eventId}/${det}: no suitable strain file found"
                    continue
                }

                $downloadUrl = $selected.download_url
                $originalBasename = ($downloadUrl -split '/')[-1]
                $rateTxt = "$($selected.sample_rate_kHz)kHz"
                $durTxt  = "$($selected.duration)s"
                Write-Log "INFO" "${eventId}/${det}: selected ${rateTxt} ${durTxt} -> $originalBasename"

                # Download to original filename, then copy to canonical name
                $originalDest = Join-Path $eventDir $originalBasename
                $canonicalDest = Join-Path $eventDir "${det}.hdf5"

                $ok = Download-File -Url $downloadUrl -Destination $originalDest

                if ($ok) {
                    # Create canonical copy (overwrite if stale)
                    if ($originalDest -ne $canonicalDest) {
                        if (Test-Path $canonicalDest) {
                            Remove-Item $canonicalDest -Force
                        }
                        Copy-Item -Path $originalDest -Destination $canonicalDest -Force
                    }

                    $fileSize = (Get-Item $canonicalDest).Length
                    $sizeMB = '{0:N1}' -f ($fileSize / 1MB)
                    $sha = Get-FileSha256 -FilePath $canonicalDest
                    Write-Log "OK" "${eventId}/${det}: ${sizeMB} MB  SHA256=$sha"
                    $eventStatus[$det] = "OK"
                }
                else {
                    $eventStatus[$det] = "ERROR"
                    Write-Log "ERROR" "${eventId}/${det}: download failed"
                }
            }
        }
        catch {
            Write-Log "ERROR" "${eventId}: $($_.Exception.Message)"
            foreach ($det in $DETECTORS) {
                if ($eventStatus[$det] -eq "PENDING") {
                    $eventStatus[$det] = "ERROR"
                }
            }
        }

        $summary[$eventId] = $eventStatus
    }

    # ========================================================================
    # Final summary
    # ========================================================================
    Write-Host ""
    Write-Host "=========================================="
    Write-Host "  DOWNLOAD SUMMARY"
    Write-Host "=========================================="

    $okCount = 0
    $missCount = 0
    $errCount = 0

    foreach ($eventId in $EVENTS) {
        $st = $summary[$eventId]
        $h1tag = $st["H1"]
        $l1tag = $st["L1"]
        Write-Host "  ${eventId}: H1=${h1tag}  L1=${l1tag}"

        foreach ($det in $DETECTORS) {
            switch ($st[$det]) {
                "OK"    { $okCount++ }
                "MISS"  { $missCount++ }
                "ERROR" { $errCount++ }
            }
        }
    }

    $total = $EVENTS.Count * $DETECTORS.Count
    Write-Host ""
    Write-Host "  Totals: $okCount OK, $missCount MISS, $errCount ERROR (of $total)"
    Write-Host ""

    # Always run self-check at the end
    Invoke-SelfCheck
}

# ============================================================================
# Entry point
# ============================================================================

if ($SelfCheckOnly) {
    Invoke-SelfCheck
}
else {
    Invoke-DownloadAll
}
