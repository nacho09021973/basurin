$ErrorActionPreference = "Stop"

$RepoRoot = (Get-Location).Path
$OutRoot  = Join-Path $RepoRoot "data\losc"
$Duration = 32
$Format   = "hdf5"

$Events = @(
  "GW150914","GW151226","GW170104","GW170608","GW170729",
  "GW170809","GW170814","GW170818","GW170823","GW190521_030229"
)

function Get-EventDetailUrl([string]$EventId) {
  $api = "https://gwosc.org/api/v2/events/$EventId"
  try {
    $resp = Invoke-RestMethod -Uri $api -Method GET
  } catch {
    throw "GWOSC request failed for ${EventId}. API=$api. $($ErrorActionPreference = "Stop"

$RepoRoot = (Get-Location).Path
$OutRoot  = Join-Path $RepoRoot "data\losc"
$Duration = 32
$Format   = "hdf5"

$Events = @(
  "GW150914","GW151226","GW170104","GW170608","GW170729",
  "GW170809","GW170814","GW170818","GW170823","GW190521_030229"
)

function Get-EventDetailUrl([string]$EventId) {
  $api = "https://gwosc.org/api/v2/events/$EventId"
  $resp = Invoke-RestMethod -Uri $api -Method GET
  $detail = $resp.events[0].versions[-1].detail_url
  if (-not $detail) { throw "No detail_url para $EventId (API=$api)" }
  return $detail
}

function Pick-DetectorUrl($StrainFilesJson, [string]$Detector, [int]$Duration, [string]$Format) {
  # StrainFilesJson suele ser un array (lista) o un objeto con propiedad 'strain_files'
  $files = $null
  if ($StrainFilesJson -is [System.Array]) {
    $files = $StrainFilesJson
  } elseif ($StrainFilesJson -and ($StrainFilesJson.PSObject.Properties.Name -contains "strain_files")) {
    $files = $StrainFilesJson.strain_files
  }

  if (-not $files) { return $null }

  # Preferencia: hdf5 + duration exacta; si no, cualquier hdf5; si no, cualquier formato.
  $cand = $files | Where-Object { $ErrorActionPreference = "Stop"

$RepoRoot = (Get-Location).Path
$OutRoot  = Join-Path $RepoRoot "data\losc"
$Duration = 32
$Format   = "hdf5"

$Events = @(
  "GW150914","GW151226","GW170104","GW170608","GW170729",
  "GW170809","GW170814","GW170818","GW170823","GW190521_030229"
)

function Get-EventDetailUrl([string]$EventId) {
  $api = "https://gwosc.org/api/v2/events/$EventId"
  try {
    $resp = Invoke-RestMethod -Uri $api -Method GET
  } catch {
    throw "GWOSC request failed for ${EventId}. API=$api. $($ErrorActionPreference = "Stop"

$RepoRoot = (Get-Location).Path
$OutRoot  = Join-Path $RepoRoot "data\losc"
$Duration = 32
$Format   = "hdf5"

$Events = @(
  "GW150914","GW151226","GW170104","GW170608","GW170729",
  "GW170809","GW170814","GW170818","GW170823","GW190521_030229"
)

function Get-EventDetailUrl([string]$EventId) {
  $api = "https://gwosc.org/api/v2/events/$EventId"
  $resp = Invoke-RestMethod -Uri $api -Method GET
  $detail = $resp.events[0].versions[-1].detail_url
  if (-not $detail) { throw "No detail_url para $EventId (API=$api)" }
  return $detail
}

function Pick-DetectorUrl($DetailJson, [string]$Detector, [int]$Duration, [string]$Format) {
  $strain = $DetailJson.strain | Where-Object { $_.detector -eq $Detector }
  if (-not $strain) { return $null }
  $file = $strain.files | Where-Object { $_.format -eq $Format -and $_.duration -eq $Duration } | Select-Object -First 1
  if (-not $file) { return $null }
  return $file.download_url
}

function Ensure-EventFiles([string]$OutDir) {
  $h1 = Get-ChildItem -Path $OutDir -Filter "*H1*.hdf5" -ErrorAction SilentlyContinue | Select-Object -First 1
  $l1 = Get-ChildItem -Path $OutDir -Filter "*L1*.hdf5" -ErrorAction SilentlyContinue | Select-Object -First 1
  return ($null -ne $h1 -and $null -ne $l1)
}

function Download-File([string]$Url, [string]$OutDir) {
  $fileName = [System.IO.Path]::GetFileName(([uri]$Url).AbsolutePath)
  $outPath = Join-Path $OutDir $fileName

  if (Test-Path $outPath) {
    $len = (Get-Item $outPath).Length
    if ($len -gt 0) {
      Write-Host "  [SKIP] Existe: $outPath ($len bytes)"
      return $outPath
    } else {
      Remove-Item -Force $outPath
    }
  }

  Write-Host "  [GET]  $Url"
  Start-BitsTransfer -Source $Url -Destination $outPath -TransferType Download -ErrorAction Stop
  return $outPath
}

New-Item -ItemType Directory -Force -Path $OutRoot | Out-Null

foreach ($EventId in $Events) {
  $OutDir  = Join-Path $OutRoot $EventId
  New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

  if (Ensure-EventFiles -OutDir $OutDir) {
    Write-Host "[OK]   ${EventId}: ya existe H1+L1 en $OutDir"
    continue
  }

  Write-Host "[MISS] ${EventId}: descargando H1+L1 (duration=$Duration format=$Format) → $OutDir"

  $detailUrl = Get-EventDetailUrl -EventId $EventId
  $detail      = Invoke-RestMethod -Uri $detailUrl -Method GET
$strainFiles = Invoke-RestMethod -Uri $detail.strain_files_url -Method GET

$h1Url = Pick-DetectorUrl -StrainFilesJson $strainFiles -Detector "H1" -Duration $Duration -Format $Format
$l1Url = Pick-DetectorUrl -StrainFilesJson $strainFiles -Detector "L1" -Duration $Duration -Format $Format

  if (-not $h1Url -or -not $l1Url) {
    throw "No encontré URLs H1/L1 para $EventId (duration=$Duration format=$Format). detail_url=$detailUrl"
  }

  $h1Path = Download-File -Url $h1Url -OutDir $OutDir
  $l1Path = Download-File -Url $l1Url -OutDir $OutDir

  Write-Host "  [HASH] $EventId"
  Get-FileHash -Algorithm SHA256 $h1Path | Format-Table -AutoSize
  Get-FileHash -Algorithm SHA256 $l1Path | Format-Table -AutoSize
}

Write-Host "[DONE] Descarga finalizada en $OutRoot"
.Exception.Message)"
  }

  # GWOSC puede devolver 'events' (lista) o 'event' (objeto) o estructuras diferentes.
  $detail = $null

  if ($resp -and $resp.PSObject.Properties.Name -contains "events" -and $resp.events -and $resp.events.Count -gt 0) {
    $ev = $resp.events[0]
    if ($ev.versions -and $ev.versions.Count -gt 0) { $detail = $ev.versions[-1].detail_url }
    if (-not $detail -and $ev.detail_url) { $detail = $ev.detail_url }
  } elseif ($resp -and $resp.PSObject.Properties.Name -contains "event" -and $resp.event) {
    $ev = $resp.event
    if ($ev.versions -and $ev.versions.Count -gt 0) { $detail = $ev.versions[-1].detail_url }
    if (-not $detail -and $ev.detail_url) { $detail = $ev.detail_url }
  } elseif ($resp -and $resp.PSObject.Properties.Name -contains "detail_url") {
    $detail = $resp.detail_url
  }

    if (-not $detail -and $resp -and ($resp.PSObject.Properties.Name -contains "versions") -and $resp.versions -and $resp.versions.Count -gt 0) {
    $detail = $resp.versions[-1].detail_url
  }

  if (-not $detail) {
    # Dump mínimo auditable (sin volcar 10k líneas)
    $keys = if ($resp) { ($resp.PSObject.Properties.Name -join ",") } else { "<null>" }
    throw "No detail_url para ${EventId}. API=$api. Top-level keys=[$keys]"
  }
  return $detail
}function Pick-DetectorUrl($DetailJson, [string]$Detector, [int]$Duration, [string]$Format) {
  $strain = $DetailJson.strain | Where-Object { $_.detector -eq $Detector }
  if (-not $strain) { return $null }
  $file = $strain.files | Where-Object { $_.format -eq $Format -and $_.duration -eq $Duration } | Select-Object -First 1
  if (-not $file) { return $null }
  return $file.download_url
}

function Ensure-EventFiles([string]$OutDir) {
  $h1 = Get-ChildItem -Path $OutDir -Filter "*H1*.hdf5" -ErrorAction SilentlyContinue | Select-Object -First 1
  $l1 = Get-ChildItem -Path $OutDir -Filter "*L1*.hdf5" -ErrorAction SilentlyContinue | Select-Object -First 1
  return ($null -ne $h1 -and $null -ne $l1)
}

function Download-File([string]$Url, [string]$OutDir) {
  $fileName = [System.IO.Path]::GetFileName(([uri]$Url).AbsolutePath)
  $outPath = Join-Path $OutDir $fileName

  if (Test-Path $outPath) {
    $len = (Get-Item $outPath).Length
    if ($len -gt 0) {
      Write-Host "  [SKIP] Existe: $outPath ($len bytes)"
      return $outPath
    } else {
      Remove-Item -Force $outPath
    }
  }

  Write-Host "  [GET]  $Url"
  Start-BitsTransfer -Source $Url -Destination $outPath -TransferType Download -ErrorAction Stop
  return $outPath
}

New-Item -ItemType Directory -Force -Path $OutRoot | Out-Null

foreach ($EventId in $Events) {
  $OutDir  = Join-Path $OutRoot $EventId
  New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

  if (Ensure-EventFiles -OutDir $OutDir) {
    Write-Host "[OK]   ${EventId}: ya existe H1+L1 en $OutDir"
    continue
  }

  Write-Host "[MISS] ${EventId}: descargando H1+L1 (duration=$Duration format=$Format) → $OutDir"

  $detailUrl = Get-EventDetailUrl -EventId $EventId
  $detail      = Invoke-RestMethod -Uri $detailUrl -Method GET
$strainFiles = Invoke-RestMethod -Uri $detail.strain_files_url -Method GET

$h1Url = Pick-DetectorUrl -StrainFilesJson $strainFiles -Detector "H1" -Duration $Duration -Format $Format
$l1Url = Pick-DetectorUrl -StrainFilesJson $strainFiles -Detector "L1" -Duration $Duration -Format $Format

  if (-not $h1Url -or -not $l1Url) {
    throw "No encontré URLs H1/L1 para $EventId (duration=$Duration format=$Format). detail_url=$detailUrl"
  }

  $h1Path = Download-File -Url $h1Url -OutDir $OutDir
  $l1Path = Download-File -Url $l1Url -OutDir $OutDir

  Write-Host "  [HASH] $EventId"
  Get-FileHash -Algorithm SHA256 $h1Path | Format-Table -AutoSize
  Get-FileHash -Algorithm SHA256 $l1Path | Format-Table -AutoSize
}

Write-Host "[DONE] Descarga finalizada en $OutRoot"


.detector -eq $Detector -and $ErrorActionPreference = "Stop"

$RepoRoot = (Get-Location).Path
$OutRoot  = Join-Path $RepoRoot "data\losc"
$Duration = 32
$Format   = "hdf5"

$Events = @(
  "GW150914","GW151226","GW170104","GW170608","GW170729",
  "GW170809","GW170814","GW170818","GW170823","GW190521_030229"
)

function Get-EventDetailUrl([string]$EventId) {
  $api = "https://gwosc.org/api/v2/events/$EventId"
  try {
    $resp = Invoke-RestMethod -Uri $api -Method GET
  } catch {
    throw "GWOSC request failed for ${EventId}. API=$api. $($ErrorActionPreference = "Stop"

$RepoRoot = (Get-Location).Path
$OutRoot  = Join-Path $RepoRoot "data\losc"
$Duration = 32
$Format   = "hdf5"

$Events = @(
  "GW150914","GW151226","GW170104","GW170608","GW170729",
  "GW170809","GW170814","GW170818","GW170823","GW190521_030229"
)

function Get-EventDetailUrl([string]$EventId) {
  $api = "https://gwosc.org/api/v2/events/$EventId"
  $resp = Invoke-RestMethod -Uri $api -Method GET
  $detail = $resp.events[0].versions[-1].detail_url
  if (-not $detail) { throw "No detail_url para $EventId (API=$api)" }
  return $detail
}

function Pick-DetectorUrl($DetailJson, [string]$Detector, [int]$Duration, [string]$Format) {
  $strain = $DetailJson.strain | Where-Object { $_.detector -eq $Detector }
  if (-not $strain) { return $null }
  $file = $strain.files | Where-Object { $_.format -eq $Format -and $_.duration -eq $Duration } | Select-Object -First 1
  if (-not $file) { return $null }
  return $file.download_url
}

function Ensure-EventFiles([string]$OutDir) {
  $h1 = Get-ChildItem -Path $OutDir -Filter "*H1*.hdf5" -ErrorAction SilentlyContinue | Select-Object -First 1
  $l1 = Get-ChildItem -Path $OutDir -Filter "*L1*.hdf5" -ErrorAction SilentlyContinue | Select-Object -First 1
  return ($null -ne $h1 -and $null -ne $l1)
}

function Download-File([string]$Url, [string]$OutDir) {
  $fileName = [System.IO.Path]::GetFileName(([uri]$Url).AbsolutePath)
  $outPath = Join-Path $OutDir $fileName

  if (Test-Path $outPath) {
    $len = (Get-Item $outPath).Length
    if ($len -gt 0) {
      Write-Host "  [SKIP] Existe: $outPath ($len bytes)"
      return $outPath
    } else {
      Remove-Item -Force $outPath
    }
  }

  Write-Host "  [GET]  $Url"
  Start-BitsTransfer -Source $Url -Destination $outPath -TransferType Download -ErrorAction Stop
  return $outPath
}

New-Item -ItemType Directory -Force -Path $OutRoot | Out-Null

foreach ($EventId in $Events) {
  $OutDir  = Join-Path $OutRoot $EventId
  New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

  if (Ensure-EventFiles -OutDir $OutDir) {
    Write-Host "[OK]   ${EventId}: ya existe H1+L1 en $OutDir"
    continue
  }

  Write-Host "[MISS] ${EventId}: descargando H1+L1 (duration=$Duration format=$Format) → $OutDir"

  $detailUrl = Get-EventDetailUrl -EventId $EventId
  $detail      = Invoke-RestMethod -Uri $detailUrl -Method GET
$strainFiles = Invoke-RestMethod -Uri $detail.strain_files_url -Method GET

$h1Url = Pick-DetectorUrl -StrainFilesJson $strainFiles -Detector "H1" -Duration $Duration -Format $Format
$l1Url = Pick-DetectorUrl -StrainFilesJson $strainFiles -Detector "L1" -Duration $Duration -Format $Format

  if (-not $h1Url -or -not $l1Url) {
    throw "No encontré URLs H1/L1 para $EventId (duration=$Duration format=$Format). detail_url=$detailUrl"
  }

  $h1Path = Download-File -Url $h1Url -OutDir $OutDir
  $l1Path = Download-File -Url $l1Url -OutDir $OutDir

  Write-Host "  [HASH] $EventId"
  Get-FileHash -Algorithm SHA256 $h1Path | Format-Table -AutoSize
  Get-FileHash -Algorithm SHA256 $l1Path | Format-Table -AutoSize
}

Write-Host "[DONE] Descarga finalizada en $OutRoot"
.Exception.Message)"
  }

  # GWOSC puede devolver 'events' (lista) o 'event' (objeto) o estructuras diferentes.
  $detail = $null

  if ($resp -and $resp.PSObject.Properties.Name -contains "events" -and $resp.events -and $resp.events.Count -gt 0) {
    $ev = $resp.events[0]
    if ($ev.versions -and $ev.versions.Count -gt 0) { $detail = $ev.versions[-1].detail_url }
    if (-not $detail -and $ev.detail_url) { $detail = $ev.detail_url }
  } elseif ($resp -and $resp.PSObject.Properties.Name -contains "event" -and $resp.event) {
    $ev = $resp.event
    if ($ev.versions -and $ev.versions.Count -gt 0) { $detail = $ev.versions[-1].detail_url }
    if (-not $detail -and $ev.detail_url) { $detail = $ev.detail_url }
  } elseif ($resp -and $resp.PSObject.Properties.Name -contains "detail_url") {
    $detail = $resp.detail_url
  }

    if (-not $detail -and $resp -and ($resp.PSObject.Properties.Name -contains "versions") -and $resp.versions -and $resp.versions.Count -gt 0) {
    $detail = $resp.versions[-1].detail_url
  }

  if (-not $detail) {
    # Dump mínimo auditable (sin volcar 10k líneas)
    $keys = if ($resp) { ($resp.PSObject.Properties.Name -join ",") } else { "<null>" }
    throw "No detail_url para ${EventId}. API=$api. Top-level keys=[$keys]"
  }
  return $detail
}function Pick-DetectorUrl($DetailJson, [string]$Detector, [int]$Duration, [string]$Format) {
  $strain = $DetailJson.strain | Where-Object { $_.detector -eq $Detector }
  if (-not $strain) { return $null }
  $file = $strain.files | Where-Object { $_.format -eq $Format -and $_.duration -eq $Duration } | Select-Object -First 1
  if (-not $file) { return $null }
  return $file.download_url
}

function Ensure-EventFiles([string]$OutDir) {
  $h1 = Get-ChildItem -Path $OutDir -Filter "*H1*.hdf5" -ErrorAction SilentlyContinue | Select-Object -First 1
  $l1 = Get-ChildItem -Path $OutDir -Filter "*L1*.hdf5" -ErrorAction SilentlyContinue | Select-Object -First 1
  return ($null -ne $h1 -and $null -ne $l1)
}

function Download-File([string]$Url, [string]$OutDir) {
  $fileName = [System.IO.Path]::GetFileName(([uri]$Url).AbsolutePath)
  $outPath = Join-Path $OutDir $fileName

  if (Test-Path $outPath) {
    $len = (Get-Item $outPath).Length
    if ($len -gt 0) {
      Write-Host "  [SKIP] Existe: $outPath ($len bytes)"
      return $outPath
    } else {
      Remove-Item -Force $outPath
    }
  }

  Write-Host "  [GET]  $Url"
  Start-BitsTransfer -Source $Url -Destination $outPath -TransferType Download -ErrorAction Stop
  return $outPath
}

New-Item -ItemType Directory -Force -Path $OutRoot | Out-Null

foreach ($EventId in $Events) {
  $OutDir  = Join-Path $OutRoot $EventId
  New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

  if (Ensure-EventFiles -OutDir $OutDir) {
    Write-Host "[OK]   ${EventId}: ya existe H1+L1 en $OutDir"
    continue
  }

  Write-Host "[MISS] ${EventId}: descargando H1+L1 (duration=$Duration format=$Format) → $OutDir"

  $detailUrl = Get-EventDetailUrl -EventId $EventId
  $detail      = Invoke-RestMethod -Uri $detailUrl -Method GET
$strainFiles = Invoke-RestMethod -Uri $detail.strain_files_url -Method GET

$h1Url = Pick-DetectorUrl -StrainFilesJson $strainFiles -Detector "H1" -Duration $Duration -Format $Format
$l1Url = Pick-DetectorUrl -StrainFilesJson $strainFiles -Detector "L1" -Duration $Duration -Format $Format

  if (-not $h1Url -or -not $l1Url) {
    throw "No encontré URLs H1/L1 para $EventId (duration=$Duration format=$Format). detail_url=$detailUrl"
  }

  $h1Path = Download-File -Url $h1Url -OutDir $OutDir
  $l1Path = Download-File -Url $l1Url -OutDir $OutDir

  Write-Host "  [HASH] $EventId"
  Get-FileHash -Algorithm SHA256 $h1Path | Format-Table -AutoSize
  Get-FileHash -Algorithm SHA256 $l1Path | Format-Table -AutoSize
}

Write-Host "[DONE] Descarga finalizada en $OutRoot"


.format -eq $Format -and $ErrorActionPreference = "Stop"

$RepoRoot = (Get-Location).Path
$OutRoot  = Join-Path $RepoRoot "data\losc"
$Duration = 32
$Format   = "hdf5"

$Events = @(
  "GW150914","GW151226","GW170104","GW170608","GW170729",
  "GW170809","GW170814","GW170818","GW170823","GW190521_030229"
)

function Get-EventDetailUrl([string]$EventId) {
  $api = "https://gwosc.org/api/v2/events/$EventId"
  try {
    $resp = Invoke-RestMethod -Uri $api -Method GET
  } catch {
    throw "GWOSC request failed for ${EventId}. API=$api. $($ErrorActionPreference = "Stop"

$RepoRoot = (Get-Location).Path
$OutRoot  = Join-Path $RepoRoot "data\losc"
$Duration = 32
$Format   = "hdf5"

$Events = @(
  "GW150914","GW151226","GW170104","GW170608","GW170729",
  "GW170809","GW170814","GW170818","GW170823","GW190521_030229"
)

function Get-EventDetailUrl([string]$EventId) {
  $api = "https://gwosc.org/api/v2/events/$EventId"
  $resp = Invoke-RestMethod -Uri $api -Method GET
  $detail = $resp.events[0].versions[-1].detail_url
  if (-not $detail) { throw "No detail_url para $EventId (API=$api)" }
  return $detail
}

function Pick-DetectorUrl($DetailJson, [string]$Detector, [int]$Duration, [string]$Format) {
  $strain = $DetailJson.strain | Where-Object { $_.detector -eq $Detector }
  if (-not $strain) { return $null }
  $file = $strain.files | Where-Object { $_.format -eq $Format -and $_.duration -eq $Duration } | Select-Object -First 1
  if (-not $file) { return $null }
  return $file.download_url
}

function Ensure-EventFiles([string]$OutDir) {
  $h1 = Get-ChildItem -Path $OutDir -Filter "*H1*.hdf5" -ErrorAction SilentlyContinue | Select-Object -First 1
  $l1 = Get-ChildItem -Path $OutDir -Filter "*L1*.hdf5" -ErrorAction SilentlyContinue | Select-Object -First 1
  return ($null -ne $h1 -and $null -ne $l1)
}

function Download-File([string]$Url, [string]$OutDir) {
  $fileName = [System.IO.Path]::GetFileName(([uri]$Url).AbsolutePath)
  $outPath = Join-Path $OutDir $fileName

  if (Test-Path $outPath) {
    $len = (Get-Item $outPath).Length
    if ($len -gt 0) {
      Write-Host "  [SKIP] Existe: $outPath ($len bytes)"
      return $outPath
    } else {
      Remove-Item -Force $outPath
    }
  }

  Write-Host "  [GET]  $Url"
  Start-BitsTransfer -Source $Url -Destination $outPath -TransferType Download -ErrorAction Stop
  return $outPath
}

New-Item -ItemType Directory -Force -Path $OutRoot | Out-Null

foreach ($EventId in $Events) {
  $OutDir  = Join-Path $OutRoot $EventId
  New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

  if (Ensure-EventFiles -OutDir $OutDir) {
    Write-Host "[OK]   ${EventId}: ya existe H1+L1 en $OutDir"
    continue
  }

  Write-Host "[MISS] ${EventId}: descargando H1+L1 (duration=$Duration format=$Format) → $OutDir"

  $detailUrl = Get-EventDetailUrl -EventId $EventId
  $detail      = Invoke-RestMethod -Uri $detailUrl -Method GET
$strainFiles = Invoke-RestMethod -Uri $detail.strain_files_url -Method GET

$h1Url = Pick-DetectorUrl -StrainFilesJson $strainFiles -Detector "H1" -Duration $Duration -Format $Format
$l1Url = Pick-DetectorUrl -StrainFilesJson $strainFiles -Detector "L1" -Duration $Duration -Format $Format

  if (-not $h1Url -or -not $l1Url) {
    throw "No encontré URLs H1/L1 para $EventId (duration=$Duration format=$Format). detail_url=$detailUrl"
  }

  $h1Path = Download-File -Url $h1Url -OutDir $OutDir
  $l1Path = Download-File -Url $l1Url -OutDir $OutDir

  Write-Host "  [HASH] $EventId"
  Get-FileHash -Algorithm SHA256 $h1Path | Format-Table -AutoSize
  Get-FileHash -Algorithm SHA256 $l1Path | Format-Table -AutoSize
}

Write-Host "[DONE] Descarga finalizada en $OutRoot"
.Exception.Message)"
  }

  # GWOSC puede devolver 'events' (lista) o 'event' (objeto) o estructuras diferentes.
  $detail = $null

  if ($resp -and $resp.PSObject.Properties.Name -contains "events" -and $resp.events -and $resp.events.Count -gt 0) {
    $ev = $resp.events[0]
    if ($ev.versions -and $ev.versions.Count -gt 0) { $detail = $ev.versions[-1].detail_url }
    if (-not $detail -and $ev.detail_url) { $detail = $ev.detail_url }
  } elseif ($resp -and $resp.PSObject.Properties.Name -contains "event" -and $resp.event) {
    $ev = $resp.event
    if ($ev.versions -and $ev.versions.Count -gt 0) { $detail = $ev.versions[-1].detail_url }
    if (-not $detail -and $ev.detail_url) { $detail = $ev.detail_url }
  } elseif ($resp -and $resp.PSObject.Properties.Name -contains "detail_url") {
    $detail = $resp.detail_url
  }

    if (-not $detail -and $resp -and ($resp.PSObject.Properties.Name -contains "versions") -and $resp.versions -and $resp.versions.Count -gt 0) {
    $detail = $resp.versions[-1].detail_url
  }

  if (-not $detail) {
    # Dump mínimo auditable (sin volcar 10k líneas)
    $keys = if ($resp) { ($resp.PSObject.Properties.Name -join ",") } else { "<null>" }
    throw "No detail_url para ${EventId}. API=$api. Top-level keys=[$keys]"
  }
  return $detail
}function Pick-DetectorUrl($DetailJson, [string]$Detector, [int]$Duration, [string]$Format) {
  $strain = $DetailJson.strain | Where-Object { $_.detector -eq $Detector }
  if (-not $strain) { return $null }
  $file = $strain.files | Where-Object { $_.format -eq $Format -and $_.duration -eq $Duration } | Select-Object -First 1
  if (-not $file) { return $null }
  return $file.download_url
}

function Ensure-EventFiles([string]$OutDir) {
  $h1 = Get-ChildItem -Path $OutDir -Filter "*H1*.hdf5" -ErrorAction SilentlyContinue | Select-Object -First 1
  $l1 = Get-ChildItem -Path $OutDir -Filter "*L1*.hdf5" -ErrorAction SilentlyContinue | Select-Object -First 1
  return ($null -ne $h1 -and $null -ne $l1)
}

function Download-File([string]$Url, [string]$OutDir) {
  $fileName = [System.IO.Path]::GetFileName(([uri]$Url).AbsolutePath)
  $outPath = Join-Path $OutDir $fileName

  if (Test-Path $outPath) {
    $len = (Get-Item $outPath).Length
    if ($len -gt 0) {
      Write-Host "  [SKIP] Existe: $outPath ($len bytes)"
      return $outPath
    } else {
      Remove-Item -Force $outPath
    }
  }

  Write-Host "  [GET]  $Url"
  Start-BitsTransfer -Source $Url -Destination $outPath -TransferType Download -ErrorAction Stop
  return $outPath
}

New-Item -ItemType Directory -Force -Path $OutRoot | Out-Null

foreach ($EventId in $Events) {
  $OutDir  = Join-Path $OutRoot $EventId
  New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

  if (Ensure-EventFiles -OutDir $OutDir) {
    Write-Host "[OK]   ${EventId}: ya existe H1+L1 en $OutDir"
    continue
  }

  Write-Host "[MISS] ${EventId}: descargando H1+L1 (duration=$Duration format=$Format) → $OutDir"

  $detailUrl = Get-EventDetailUrl -EventId $EventId
  $detail      = Invoke-RestMethod -Uri $detailUrl -Method GET
$strainFiles = Invoke-RestMethod -Uri $detail.strain_files_url -Method GET

$h1Url = Pick-DetectorUrl -StrainFilesJson $strainFiles -Detector "H1" -Duration $Duration -Format $Format
$l1Url = Pick-DetectorUrl -StrainFilesJson $strainFiles -Detector "L1" -Duration $Duration -Format $Format

  if (-not $h1Url -or -not $l1Url) {
    throw "No encontré URLs H1/L1 para $EventId (duration=$Duration format=$Format). detail_url=$detailUrl"
  }

  $h1Path = Download-File -Url $h1Url -OutDir $OutDir
  $l1Path = Download-File -Url $l1Url -OutDir $OutDir

  Write-Host "  [HASH] $EventId"
  Get-FileHash -Algorithm SHA256 $h1Path | Format-Table -AutoSize
  Get-FileHash -Algorithm SHA256 $l1Path | Format-Table -AutoSize
}

Write-Host "[DONE] Descarga finalizada en $OutRoot"


.duration -eq $Duration } | Select-Object -First 1
  if ($cand -and $cand.download_url) { return $cand.download_url }

  $cand = $files | Where-Object { $ErrorActionPreference = "Stop"

$RepoRoot = (Get-Location).Path
$OutRoot  = Join-Path $RepoRoot "data\losc"
$Duration = 32
$Format   = "hdf5"

$Events = @(
  "GW150914","GW151226","GW170104","GW170608","GW170729",
  "GW170809","GW170814","GW170818","GW170823","GW190521_030229"
)

function Get-EventDetailUrl([string]$EventId) {
  $api = "https://gwosc.org/api/v2/events/$EventId"
  try {
    $resp = Invoke-RestMethod -Uri $api -Method GET
  } catch {
    throw "GWOSC request failed for ${EventId}. API=$api. $($ErrorActionPreference = "Stop"

$RepoRoot = (Get-Location).Path
$OutRoot  = Join-Path $RepoRoot "data\losc"
$Duration = 32
$Format   = "hdf5"

$Events = @(
  "GW150914","GW151226","GW170104","GW170608","GW170729",
  "GW170809","GW170814","GW170818","GW170823","GW190521_030229"
)

function Get-EventDetailUrl([string]$EventId) {
  $api = "https://gwosc.org/api/v2/events/$EventId"
  $resp = Invoke-RestMethod -Uri $api -Method GET
  $detail = $resp.events[0].versions[-1].detail_url
  if (-not $detail) { throw "No detail_url para $EventId (API=$api)" }
  return $detail
}

function Pick-DetectorUrl($DetailJson, [string]$Detector, [int]$Duration, [string]$Format) {
  $strain = $DetailJson.strain | Where-Object { $_.detector -eq $Detector }
  if (-not $strain) { return $null }
  $file = $strain.files | Where-Object { $_.format -eq $Format -and $_.duration -eq $Duration } | Select-Object -First 1
  if (-not $file) { return $null }
  return $file.download_url
}

function Ensure-EventFiles([string]$OutDir) {
  $h1 = Get-ChildItem -Path $OutDir -Filter "*H1*.hdf5" -ErrorAction SilentlyContinue | Select-Object -First 1
  $l1 = Get-ChildItem -Path $OutDir -Filter "*L1*.hdf5" -ErrorAction SilentlyContinue | Select-Object -First 1
  return ($null -ne $h1 -and $null -ne $l1)
}

function Download-File([string]$Url, [string]$OutDir) {
  $fileName = [System.IO.Path]::GetFileName(([uri]$Url).AbsolutePath)
  $outPath = Join-Path $OutDir $fileName

  if (Test-Path $outPath) {
    $len = (Get-Item $outPath).Length
    if ($len -gt 0) {
      Write-Host "  [SKIP] Existe: $outPath ($len bytes)"
      return $outPath
    } else {
      Remove-Item -Force $outPath
    }
  }

  Write-Host "  [GET]  $Url"
  Start-BitsTransfer -Source $Url -Destination $outPath -TransferType Download -ErrorAction Stop
  return $outPath
}

New-Item -ItemType Directory -Force -Path $OutRoot | Out-Null

foreach ($EventId in $Events) {
  $OutDir  = Join-Path $OutRoot $EventId
  New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

  if (Ensure-EventFiles -OutDir $OutDir) {
    Write-Host "[OK]   ${EventId}: ya existe H1+L1 en $OutDir"
    continue
  }

  Write-Host "[MISS] ${EventId}: descargando H1+L1 (duration=$Duration format=$Format) → $OutDir"

  $detailUrl = Get-EventDetailUrl -EventId $EventId
  $detail      = Invoke-RestMethod -Uri $detailUrl -Method GET
$strainFiles = Invoke-RestMethod -Uri $detail.strain_files_url -Method GET

$h1Url = Pick-DetectorUrl -StrainFilesJson $strainFiles -Detector "H1" -Duration $Duration -Format $Format
$l1Url = Pick-DetectorUrl -StrainFilesJson $strainFiles -Detector "L1" -Duration $Duration -Format $Format

  if (-not $h1Url -or -not $l1Url) {
    throw "No encontré URLs H1/L1 para $EventId (duration=$Duration format=$Format). detail_url=$detailUrl"
  }

  $h1Path = Download-File -Url $h1Url -OutDir $OutDir
  $l1Path = Download-File -Url $l1Url -OutDir $OutDir

  Write-Host "  [HASH] $EventId"
  Get-FileHash -Algorithm SHA256 $h1Path | Format-Table -AutoSize
  Get-FileHash -Algorithm SHA256 $l1Path | Format-Table -AutoSize
}

Write-Host "[DONE] Descarga finalizada en $OutRoot"
.Exception.Message)"
  }

  # GWOSC puede devolver 'events' (lista) o 'event' (objeto) o estructuras diferentes.
  $detail = $null

  if ($resp -and $resp.PSObject.Properties.Name -contains "events" -and $resp.events -and $resp.events.Count -gt 0) {
    $ev = $resp.events[0]
    if ($ev.versions -and $ev.versions.Count -gt 0) { $detail = $ev.versions[-1].detail_url }
    if (-not $detail -and $ev.detail_url) { $detail = $ev.detail_url }
  } elseif ($resp -and $resp.PSObject.Properties.Name -contains "event" -and $resp.event) {
    $ev = $resp.event
    if ($ev.versions -and $ev.versions.Count -gt 0) { $detail = $ev.versions[-1].detail_url }
    if (-not $detail -and $ev.detail_url) { $detail = $ev.detail_url }
  } elseif ($resp -and $resp.PSObject.Properties.Name -contains "detail_url") {
    $detail = $resp.detail_url
  }

    if (-not $detail -and $resp -and ($resp.PSObject.Properties.Name -contains "versions") -and $resp.versions -and $resp.versions.Count -gt 0) {
    $detail = $resp.versions[-1].detail_url
  }

  if (-not $detail) {
    # Dump mínimo auditable (sin volcar 10k líneas)
    $keys = if ($resp) { ($resp.PSObject.Properties.Name -join ",") } else { "<null>" }
    throw "No detail_url para ${EventId}. API=$api. Top-level keys=[$keys]"
  }
  return $detail
}function Pick-DetectorUrl($DetailJson, [string]$Detector, [int]$Duration, [string]$Format) {
  $strain = $DetailJson.strain | Where-Object { $_.detector -eq $Detector }
  if (-not $strain) { return $null }
  $file = $strain.files | Where-Object { $_.format -eq $Format -and $_.duration -eq $Duration } | Select-Object -First 1
  if (-not $file) { return $null }
  return $file.download_url
}

function Ensure-EventFiles([string]$OutDir) {
  $h1 = Get-ChildItem -Path $OutDir -Filter "*H1*.hdf5" -ErrorAction SilentlyContinue | Select-Object -First 1
  $l1 = Get-ChildItem -Path $OutDir -Filter "*L1*.hdf5" -ErrorAction SilentlyContinue | Select-Object -First 1
  return ($null -ne $h1 -and $null -ne $l1)
}

function Download-File([string]$Url, [string]$OutDir) {
  $fileName = [System.IO.Path]::GetFileName(([uri]$Url).AbsolutePath)
  $outPath = Join-Path $OutDir $fileName

  if (Test-Path $outPath) {
    $len = (Get-Item $outPath).Length
    if ($len -gt 0) {
      Write-Host "  [SKIP] Existe: $outPath ($len bytes)"
      return $outPath
    } else {
      Remove-Item -Force $outPath
    }
  }

  Write-Host "  [GET]  $Url"
  Start-BitsTransfer -Source $Url -Destination $outPath -TransferType Download -ErrorAction Stop
  return $outPath
}

New-Item -ItemType Directory -Force -Path $OutRoot | Out-Null

foreach ($EventId in $Events) {
  $OutDir  = Join-Path $OutRoot $EventId
  New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

  if (Ensure-EventFiles -OutDir $OutDir) {
    Write-Host "[OK]   ${EventId}: ya existe H1+L1 en $OutDir"
    continue
  }

  Write-Host "[MISS] ${EventId}: descargando H1+L1 (duration=$Duration format=$Format) → $OutDir"

  $detailUrl = Get-EventDetailUrl -EventId $EventId
  $detail      = Invoke-RestMethod -Uri $detailUrl -Method GET
$strainFiles = Invoke-RestMethod -Uri $detail.strain_files_url -Method GET

$h1Url = Pick-DetectorUrl -StrainFilesJson $strainFiles -Detector "H1" -Duration $Duration -Format $Format
$l1Url = Pick-DetectorUrl -StrainFilesJson $strainFiles -Detector "L1" -Duration $Duration -Format $Format

  if (-not $h1Url -or -not $l1Url) {
    throw "No encontré URLs H1/L1 para $EventId (duration=$Duration format=$Format). detail_url=$detailUrl"
  }

  $h1Path = Download-File -Url $h1Url -OutDir $OutDir
  $l1Path = Download-File -Url $l1Url -OutDir $OutDir

  Write-Host "  [HASH] $EventId"
  Get-FileHash -Algorithm SHA256 $h1Path | Format-Table -AutoSize
  Get-FileHash -Algorithm SHA256 $l1Path | Format-Table -AutoSize
}

Write-Host "[DONE] Descarga finalizada en $OutRoot"


.detector -eq $Detector -and $ErrorActionPreference = "Stop"

$RepoRoot = (Get-Location).Path
$OutRoot  = Join-Path $RepoRoot "data\losc"
$Duration = 32
$Format   = "hdf5"

$Events = @(
  "GW150914","GW151226","GW170104","GW170608","GW170729",
  "GW170809","GW170814","GW170818","GW170823","GW190521_030229"
)

function Get-EventDetailUrl([string]$EventId) {
  $api = "https://gwosc.org/api/v2/events/$EventId"
  try {
    $resp = Invoke-RestMethod -Uri $api -Method GET
  } catch {
    throw "GWOSC request failed for ${EventId}. API=$api. $($ErrorActionPreference = "Stop"

$RepoRoot = (Get-Location).Path
$OutRoot  = Join-Path $RepoRoot "data\losc"
$Duration = 32
$Format   = "hdf5"

$Events = @(
  "GW150914","GW151226","GW170104","GW170608","GW170729",
  "GW170809","GW170814","GW170818","GW170823","GW190521_030229"
)

function Get-EventDetailUrl([string]$EventId) {
  $api = "https://gwosc.org/api/v2/events/$EventId"
  $resp = Invoke-RestMethod -Uri $api -Method GET
  $detail = $resp.events[0].versions[-1].detail_url
  if (-not $detail) { throw "No detail_url para $EventId (API=$api)" }
  return $detail
}

function Pick-DetectorUrl($DetailJson, [string]$Detector, [int]$Duration, [string]$Format) {
  $strain = $DetailJson.strain | Where-Object { $_.detector -eq $Detector }
  if (-not $strain) { return $null }
  $file = $strain.files | Where-Object { $_.format -eq $Format -and $_.duration -eq $Duration } | Select-Object -First 1
  if (-not $file) { return $null }
  return $file.download_url
}

function Ensure-EventFiles([string]$OutDir) {
  $h1 = Get-ChildItem -Path $OutDir -Filter "*H1*.hdf5" -ErrorAction SilentlyContinue | Select-Object -First 1
  $l1 = Get-ChildItem -Path $OutDir -Filter "*L1*.hdf5" -ErrorAction SilentlyContinue | Select-Object -First 1
  return ($null -ne $h1 -and $null -ne $l1)
}

function Download-File([string]$Url, [string]$OutDir) {
  $fileName = [System.IO.Path]::GetFileName(([uri]$Url).AbsolutePath)
  $outPath = Join-Path $OutDir $fileName

  if (Test-Path $outPath) {
    $len = (Get-Item $outPath).Length
    if ($len -gt 0) {
      Write-Host "  [SKIP] Existe: $outPath ($len bytes)"
      return $outPath
    } else {
      Remove-Item -Force $outPath
    }
  }

  Write-Host "  [GET]  $Url"
  Start-BitsTransfer -Source $Url -Destination $outPath -TransferType Download -ErrorAction Stop
  return $outPath
}

New-Item -ItemType Directory -Force -Path $OutRoot | Out-Null

foreach ($EventId in $Events) {
  $OutDir  = Join-Path $OutRoot $EventId
  New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

  if (Ensure-EventFiles -OutDir $OutDir) {
    Write-Host "[OK]   ${EventId}: ya existe H1+L1 en $OutDir"
    continue
  }

  Write-Host "[MISS] ${EventId}: descargando H1+L1 (duration=$Duration format=$Format) → $OutDir"

  $detailUrl = Get-EventDetailUrl -EventId $EventId
  $detail      = Invoke-RestMethod -Uri $detailUrl -Method GET
$strainFiles = Invoke-RestMethod -Uri $detail.strain_files_url -Method GET

$h1Url = Pick-DetectorUrl -StrainFilesJson $strainFiles -Detector "H1" -Duration $Duration -Format $Format
$l1Url = Pick-DetectorUrl -StrainFilesJson $strainFiles -Detector "L1" -Duration $Duration -Format $Format

  if (-not $h1Url -or -not $l1Url) {
    throw "No encontré URLs H1/L1 para $EventId (duration=$Duration format=$Format). detail_url=$detailUrl"
  }

  $h1Path = Download-File -Url $h1Url -OutDir $OutDir
  $l1Path = Download-File -Url $l1Url -OutDir $OutDir

  Write-Host "  [HASH] $EventId"
  Get-FileHash -Algorithm SHA256 $h1Path | Format-Table -AutoSize
  Get-FileHash -Algorithm SHA256 $l1Path | Format-Table -AutoSize
}

Write-Host "[DONE] Descarga finalizada en $OutRoot"
.Exception.Message)"
  }

  # GWOSC puede devolver 'events' (lista) o 'event' (objeto) o estructuras diferentes.
  $detail = $null

  if ($resp -and $resp.PSObject.Properties.Name -contains "events" -and $resp.events -and $resp.events.Count -gt 0) {
    $ev = $resp.events[0]
    if ($ev.versions -and $ev.versions.Count -gt 0) { $detail = $ev.versions[-1].detail_url }
    if (-not $detail -and $ev.detail_url) { $detail = $ev.detail_url }
  } elseif ($resp -and $resp.PSObject.Properties.Name -contains "event" -and $resp.event) {
    $ev = $resp.event
    if ($ev.versions -and $ev.versions.Count -gt 0) { $detail = $ev.versions[-1].detail_url }
    if (-not $detail -and $ev.detail_url) { $detail = $ev.detail_url }
  } elseif ($resp -and $resp.PSObject.Properties.Name -contains "detail_url") {
    $detail = $resp.detail_url
  }

    if (-not $detail -and $resp -and ($resp.PSObject.Properties.Name -contains "versions") -and $resp.versions -and $resp.versions.Count -gt 0) {
    $detail = $resp.versions[-1].detail_url
  }

  if (-not $detail) {
    # Dump mínimo auditable (sin volcar 10k líneas)
    $keys = if ($resp) { ($resp.PSObject.Properties.Name -join ",") } else { "<null>" }
    throw "No detail_url para ${EventId}. API=$api. Top-level keys=[$keys]"
  }
  return $detail
}function Pick-DetectorUrl($DetailJson, [string]$Detector, [int]$Duration, [string]$Format) {
  $strain = $DetailJson.strain | Where-Object { $_.detector -eq $Detector }
  if (-not $strain) { return $null }
  $file = $strain.files | Where-Object { $_.format -eq $Format -and $_.duration -eq $Duration } | Select-Object -First 1
  if (-not $file) { return $null }
  return $file.download_url
}

function Ensure-EventFiles([string]$OutDir) {
  $h1 = Get-ChildItem -Path $OutDir -Filter "*H1*.hdf5" -ErrorAction SilentlyContinue | Select-Object -First 1
  $l1 = Get-ChildItem -Path $OutDir -Filter "*L1*.hdf5" -ErrorAction SilentlyContinue | Select-Object -First 1
  return ($null -ne $h1 -and $null -ne $l1)
}

function Download-File([string]$Url, [string]$OutDir) {
  $fileName = [System.IO.Path]::GetFileName(([uri]$Url).AbsolutePath)
  $outPath = Join-Path $OutDir $fileName

  if (Test-Path $outPath) {
    $len = (Get-Item $outPath).Length
    if ($len -gt 0) {
      Write-Host "  [SKIP] Existe: $outPath ($len bytes)"
      return $outPath
    } else {
      Remove-Item -Force $outPath
    }
  }

  Write-Host "  [GET]  $Url"
  Start-BitsTransfer -Source $Url -Destination $outPath -TransferType Download -ErrorAction Stop
  return $outPath
}

New-Item -ItemType Directory -Force -Path $OutRoot | Out-Null

foreach ($EventId in $Events) {
  $OutDir  = Join-Path $OutRoot $EventId
  New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

  if (Ensure-EventFiles -OutDir $OutDir) {
    Write-Host "[OK]   ${EventId}: ya existe H1+L1 en $OutDir"
    continue
  }

  Write-Host "[MISS] ${EventId}: descargando H1+L1 (duration=$Duration format=$Format) → $OutDir"

  $detailUrl = Get-EventDetailUrl -EventId $EventId
  $detail      = Invoke-RestMethod -Uri $detailUrl -Method GET
$strainFiles = Invoke-RestMethod -Uri $detail.strain_files_url -Method GET

$h1Url = Pick-DetectorUrl -StrainFilesJson $strainFiles -Detector "H1" -Duration $Duration -Format $Format
$l1Url = Pick-DetectorUrl -StrainFilesJson $strainFiles -Detector "L1" -Duration $Duration -Format $Format

  if (-not $h1Url -or -not $l1Url) {
    throw "No encontré URLs H1/L1 para $EventId (duration=$Duration format=$Format). detail_url=$detailUrl"
  }

  $h1Path = Download-File -Url $h1Url -OutDir $OutDir
  $l1Path = Download-File -Url $l1Url -OutDir $OutDir

  Write-Host "  [HASH] $EventId"
  Get-FileHash -Algorithm SHA256 $h1Path | Format-Table -AutoSize
  Get-FileHash -Algorithm SHA256 $l1Path | Format-Table -AutoSize
}

Write-Host "[DONE] Descarga finalizada en $OutRoot"


.format -eq $Format } | Sort-Object duration | Select-Object -First 1
  if ($cand -and $cand.download_url) { return $cand.download_url }

  $cand = $files | Where-Object { $ErrorActionPreference = "Stop"

$RepoRoot = (Get-Location).Path
$OutRoot  = Join-Path $RepoRoot "data\losc"
$Duration = 32
$Format   = "hdf5"

$Events = @(
  "GW150914","GW151226","GW170104","GW170608","GW170729",
  "GW170809","GW170814","GW170818","GW170823","GW190521_030229"
)

function Get-EventDetailUrl([string]$EventId) {
  $api = "https://gwosc.org/api/v2/events/$EventId"
  try {
    $resp = Invoke-RestMethod -Uri $api -Method GET
  } catch {
    throw "GWOSC request failed for ${EventId}. API=$api. $($ErrorActionPreference = "Stop"

$RepoRoot = (Get-Location).Path
$OutRoot  = Join-Path $RepoRoot "data\losc"
$Duration = 32
$Format   = "hdf5"

$Events = @(
  "GW150914","GW151226","GW170104","GW170608","GW170729",
  "GW170809","GW170814","GW170818","GW170823","GW190521_030229"
)

function Get-EventDetailUrl([string]$EventId) {
  $api = "https://gwosc.org/api/v2/events/$EventId"
  $resp = Invoke-RestMethod -Uri $api -Method GET
  $detail = $resp.events[0].versions[-1].detail_url
  if (-not $detail) { throw "No detail_url para $EventId (API=$api)" }
  return $detail
}

function Pick-DetectorUrl($DetailJson, [string]$Detector, [int]$Duration, [string]$Format) {
  $strain = $DetailJson.strain | Where-Object { $_.detector -eq $Detector }
  if (-not $strain) { return $null }
  $file = $strain.files | Where-Object { $_.format -eq $Format -and $_.duration -eq $Duration } | Select-Object -First 1
  if (-not $file) { return $null }
  return $file.download_url
}

function Ensure-EventFiles([string]$OutDir) {
  $h1 = Get-ChildItem -Path $OutDir -Filter "*H1*.hdf5" -ErrorAction SilentlyContinue | Select-Object -First 1
  $l1 = Get-ChildItem -Path $OutDir -Filter "*L1*.hdf5" -ErrorAction SilentlyContinue | Select-Object -First 1
  return ($null -ne $h1 -and $null -ne $l1)
}

function Download-File([string]$Url, [string]$OutDir) {
  $fileName = [System.IO.Path]::GetFileName(([uri]$Url).AbsolutePath)
  $outPath = Join-Path $OutDir $fileName

  if (Test-Path $outPath) {
    $len = (Get-Item $outPath).Length
    if ($len -gt 0) {
      Write-Host "  [SKIP] Existe: $outPath ($len bytes)"
      return $outPath
    } else {
      Remove-Item -Force $outPath
    }
  }

  Write-Host "  [GET]  $Url"
  Start-BitsTransfer -Source $Url -Destination $outPath -TransferType Download -ErrorAction Stop
  return $outPath
}

New-Item -ItemType Directory -Force -Path $OutRoot | Out-Null

foreach ($EventId in $Events) {
  $OutDir  = Join-Path $OutRoot $EventId
  New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

  if (Ensure-EventFiles -OutDir $OutDir) {
    Write-Host "[OK]   ${EventId}: ya existe H1+L1 en $OutDir"
    continue
  }

  Write-Host "[MISS] ${EventId}: descargando H1+L1 (duration=$Duration format=$Format) → $OutDir"

  $detailUrl = Get-EventDetailUrl -EventId $EventId
  $detail      = Invoke-RestMethod -Uri $detailUrl -Method GET
$strainFiles = Invoke-RestMethod -Uri $detail.strain_files_url -Method GET

$h1Url = Pick-DetectorUrl -StrainFilesJson $strainFiles -Detector "H1" -Duration $Duration -Format $Format
$l1Url = Pick-DetectorUrl -StrainFilesJson $strainFiles -Detector "L1" -Duration $Duration -Format $Format

  if (-not $h1Url -or -not $l1Url) {
    throw "No encontré URLs H1/L1 para $EventId (duration=$Duration format=$Format). detail_url=$detailUrl"
  }

  $h1Path = Download-File -Url $h1Url -OutDir $OutDir
  $l1Path = Download-File -Url $l1Url -OutDir $OutDir

  Write-Host "  [HASH] $EventId"
  Get-FileHash -Algorithm SHA256 $h1Path | Format-Table -AutoSize
  Get-FileHash -Algorithm SHA256 $l1Path | Format-Table -AutoSize
}

Write-Host "[DONE] Descarga finalizada en $OutRoot"
.Exception.Message)"
  }

  # GWOSC puede devolver 'events' (lista) o 'event' (objeto) o estructuras diferentes.
  $detail = $null

  if ($resp -and $resp.PSObject.Properties.Name -contains "events" -and $resp.events -and $resp.events.Count -gt 0) {
    $ev = $resp.events[0]
    if ($ev.versions -and $ev.versions.Count -gt 0) { $detail = $ev.versions[-1].detail_url }
    if (-not $detail -and $ev.detail_url) { $detail = $ev.detail_url }
  } elseif ($resp -and $resp.PSObject.Properties.Name -contains "event" -and $resp.event) {
    $ev = $resp.event
    if ($ev.versions -and $ev.versions.Count -gt 0) { $detail = $ev.versions[-1].detail_url }
    if (-not $detail -and $ev.detail_url) { $detail = $ev.detail_url }
  } elseif ($resp -and $resp.PSObject.Properties.Name -contains "detail_url") {
    $detail = $resp.detail_url
  }

    if (-not $detail -and $resp -and ($resp.PSObject.Properties.Name -contains "versions") -and $resp.versions -and $resp.versions.Count -gt 0) {
    $detail = $resp.versions[-1].detail_url
  }

  if (-not $detail) {
    # Dump mínimo auditable (sin volcar 10k líneas)
    $keys = if ($resp) { ($resp.PSObject.Properties.Name -join ",") } else { "<null>" }
    throw "No detail_url para ${EventId}. API=$api. Top-level keys=[$keys]"
  }
  return $detail
}function Pick-DetectorUrl($DetailJson, [string]$Detector, [int]$Duration, [string]$Format) {
  $strain = $DetailJson.strain | Where-Object { $_.detector -eq $Detector }
  if (-not $strain) { return $null }
  $file = $strain.files | Where-Object { $_.format -eq $Format -and $_.duration -eq $Duration } | Select-Object -First 1
  if (-not $file) { return $null }
  return $file.download_url
}

function Ensure-EventFiles([string]$OutDir) {
  $h1 = Get-ChildItem -Path $OutDir -Filter "*H1*.hdf5" -ErrorAction SilentlyContinue | Select-Object -First 1
  $l1 = Get-ChildItem -Path $OutDir -Filter "*L1*.hdf5" -ErrorAction SilentlyContinue | Select-Object -First 1
  return ($null -ne $h1 -and $null -ne $l1)
}

function Download-File([string]$Url, [string]$OutDir) {
  $fileName = [System.IO.Path]::GetFileName(([uri]$Url).AbsolutePath)
  $outPath = Join-Path $OutDir $fileName

  if (Test-Path $outPath) {
    $len = (Get-Item $outPath).Length
    if ($len -gt 0) {
      Write-Host "  [SKIP] Existe: $outPath ($len bytes)"
      return $outPath
    } else {
      Remove-Item -Force $outPath
    }
  }

  Write-Host "  [GET]  $Url"
  Start-BitsTransfer -Source $Url -Destination $outPath -TransferType Download -ErrorAction Stop
  return $outPath
}

New-Item -ItemType Directory -Force -Path $OutRoot | Out-Null

foreach ($EventId in $Events) {
  $OutDir  = Join-Path $OutRoot $EventId
  New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

  if (Ensure-EventFiles -OutDir $OutDir) {
    Write-Host "[OK]   ${EventId}: ya existe H1+L1 en $OutDir"
    continue
  }

  Write-Host "[MISS] ${EventId}: descargando H1+L1 (duration=$Duration format=$Format) → $OutDir"

  $detailUrl = Get-EventDetailUrl -EventId $EventId
  $detail      = Invoke-RestMethod -Uri $detailUrl -Method GET
$strainFiles = Invoke-RestMethod -Uri $detail.strain_files_url -Method GET

$h1Url = Pick-DetectorUrl -StrainFilesJson $strainFiles -Detector "H1" -Duration $Duration -Format $Format
$l1Url = Pick-DetectorUrl -StrainFilesJson $strainFiles -Detector "L1" -Duration $Duration -Format $Format

  if (-not $h1Url -or -not $l1Url) {
    throw "No encontré URLs H1/L1 para $EventId (duration=$Duration format=$Format). detail_url=$detailUrl"
  }

  $h1Path = Download-File -Url $h1Url -OutDir $OutDir
  $l1Path = Download-File -Url $l1Url -OutDir $OutDir

  Write-Host "  [HASH] $EventId"
  Get-FileHash -Algorithm SHA256 $h1Path | Format-Table -AutoSize
  Get-FileHash -Algorithm SHA256 $l1Path | Format-Table -AutoSize
}

Write-Host "[DONE] Descarga finalizada en $OutRoot"


.detector -eq $Detector -and $ErrorActionPreference = "Stop"

$RepoRoot = (Get-Location).Path
$OutRoot  = Join-Path $RepoRoot "data\losc"
$Duration = 32
$Format   = "hdf5"

$Events = @(
  "GW150914","GW151226","GW170104","GW170608","GW170729",
  "GW170809","GW170814","GW170818","GW170823","GW190521_030229"
)

function Get-EventDetailUrl([string]$EventId) {
  $api = "https://gwosc.org/api/v2/events/$EventId"
  try {
    $resp = Invoke-RestMethod -Uri $api -Method GET
  } catch {
    throw "GWOSC request failed for ${EventId}. API=$api. $($ErrorActionPreference = "Stop"

$RepoRoot = (Get-Location).Path
$OutRoot  = Join-Path $RepoRoot "data\losc"
$Duration = 32
$Format   = "hdf5"

$Events = @(
  "GW150914","GW151226","GW170104","GW170608","GW170729",
  "GW170809","GW170814","GW170818","GW170823","GW190521_030229"
)

function Get-EventDetailUrl([string]$EventId) {
  $api = "https://gwosc.org/api/v2/events/$EventId"
  $resp = Invoke-RestMethod -Uri $api -Method GET
  $detail = $resp.events[0].versions[-1].detail_url
  if (-not $detail) { throw "No detail_url para $EventId (API=$api)" }
  return $detail
}

function Pick-DetectorUrl($DetailJson, [string]$Detector, [int]$Duration, [string]$Format) {
  $strain = $DetailJson.strain | Where-Object { $_.detector -eq $Detector }
  if (-not $strain) { return $null }
  $file = $strain.files | Where-Object { $_.format -eq $Format -and $_.duration -eq $Duration } | Select-Object -First 1
  if (-not $file) { return $null }
  return $file.download_url
}

function Ensure-EventFiles([string]$OutDir) {
  $h1 = Get-ChildItem -Path $OutDir -Filter "*H1*.hdf5" -ErrorAction SilentlyContinue | Select-Object -First 1
  $l1 = Get-ChildItem -Path $OutDir -Filter "*L1*.hdf5" -ErrorAction SilentlyContinue | Select-Object -First 1
  return ($null -ne $h1 -and $null -ne $l1)
}

function Download-File([string]$Url, [string]$OutDir) {
  $fileName = [System.IO.Path]::GetFileName(([uri]$Url).AbsolutePath)
  $outPath = Join-Path $OutDir $fileName

  if (Test-Path $outPath) {
    $len = (Get-Item $outPath).Length
    if ($len -gt 0) {
      Write-Host "  [SKIP] Existe: $outPath ($len bytes)"
      return $outPath
    } else {
      Remove-Item -Force $outPath
    }
  }

  Write-Host "  [GET]  $Url"
  Start-BitsTransfer -Source $Url -Destination $outPath -TransferType Download -ErrorAction Stop
  return $outPath
}

New-Item -ItemType Directory -Force -Path $OutRoot | Out-Null

foreach ($EventId in $Events) {
  $OutDir  = Join-Path $OutRoot $EventId
  New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

  if (Ensure-EventFiles -OutDir $OutDir) {
    Write-Host "[OK]   ${EventId}: ya existe H1+L1 en $OutDir"
    continue
  }

  Write-Host "[MISS] ${EventId}: descargando H1+L1 (duration=$Duration format=$Format) → $OutDir"

  $detailUrl = Get-EventDetailUrl -EventId $EventId
  $detail      = Invoke-RestMethod -Uri $detailUrl -Method GET
$strainFiles = Invoke-RestMethod -Uri $detail.strain_files_url -Method GET

$h1Url = Pick-DetectorUrl -StrainFilesJson $strainFiles -Detector "H1" -Duration $Duration -Format $Format
$l1Url = Pick-DetectorUrl -StrainFilesJson $strainFiles -Detector "L1" -Duration $Duration -Format $Format

  if (-not $h1Url -or -not $l1Url) {
    throw "No encontré URLs H1/L1 para $EventId (duration=$Duration format=$Format). detail_url=$detailUrl"
  }

  $h1Path = Download-File -Url $h1Url -OutDir $OutDir
  $l1Path = Download-File -Url $l1Url -OutDir $OutDir

  Write-Host "  [HASH] $EventId"
  Get-FileHash -Algorithm SHA256 $h1Path | Format-Table -AutoSize
  Get-FileHash -Algorithm SHA256 $l1Path | Format-Table -AutoSize
}

Write-Host "[DONE] Descarga finalizada en $OutRoot"
.Exception.Message)"
  }

  # GWOSC puede devolver 'events' (lista) o 'event' (objeto) o estructuras diferentes.
  $detail = $null

  if ($resp -and $resp.PSObject.Properties.Name -contains "events" -and $resp.events -and $resp.events.Count -gt 0) {
    $ev = $resp.events[0]
    if ($ev.versions -and $ev.versions.Count -gt 0) { $detail = $ev.versions[-1].detail_url }
    if (-not $detail -and $ev.detail_url) { $detail = $ev.detail_url }
  } elseif ($resp -and $resp.PSObject.Properties.Name -contains "event" -and $resp.event) {
    $ev = $resp.event
    if ($ev.versions -and $ev.versions.Count -gt 0) { $detail = $ev.versions[-1].detail_url }
    if (-not $detail -and $ev.detail_url) { $detail = $ev.detail_url }
  } elseif ($resp -and $resp.PSObject.Properties.Name -contains "detail_url") {
    $detail = $resp.detail_url
  }

    if (-not $detail -and $resp -and ($resp.PSObject.Properties.Name -contains "versions") -and $resp.versions -and $resp.versions.Count -gt 0) {
    $detail = $resp.versions[-1].detail_url
  }

  if (-not $detail) {
    # Dump mínimo auditable (sin volcar 10k líneas)
    $keys = if ($resp) { ($resp.PSObject.Properties.Name -join ",") } else { "<null>" }
    throw "No detail_url para ${EventId}. API=$api. Top-level keys=[$keys]"
  }
  return $detail
}function Pick-DetectorUrl($DetailJson, [string]$Detector, [int]$Duration, [string]$Format) {
  $strain = $DetailJson.strain | Where-Object { $_.detector -eq $Detector }
  if (-not $strain) { return $null }
  $file = $strain.files | Where-Object { $_.format -eq $Format -and $_.duration -eq $Duration } | Select-Object -First 1
  if (-not $file) { return $null }
  return $file.download_url
}

function Ensure-EventFiles([string]$OutDir) {
  $h1 = Get-ChildItem -Path $OutDir -Filter "*H1*.hdf5" -ErrorAction SilentlyContinue | Select-Object -First 1
  $l1 = Get-ChildItem -Path $OutDir -Filter "*L1*.hdf5" -ErrorAction SilentlyContinue | Select-Object -First 1
  return ($null -ne $h1 -and $null -ne $l1)
}

function Download-File([string]$Url, [string]$OutDir) {
  $fileName = [System.IO.Path]::GetFileName(([uri]$Url).AbsolutePath)
  $outPath = Join-Path $OutDir $fileName

  if (Test-Path $outPath) {
    $len = (Get-Item $outPath).Length
    if ($len -gt 0) {
      Write-Host "  [SKIP] Existe: $outPath ($len bytes)"
      return $outPath
    } else {
      Remove-Item -Force $outPath
    }
  }

  Write-Host "  [GET]  $Url"
  Start-BitsTransfer -Source $Url -Destination $outPath -TransferType Download -ErrorAction Stop
  return $outPath
}

New-Item -ItemType Directory -Force -Path $OutRoot | Out-Null

foreach ($EventId in $Events) {
  $OutDir  = Join-Path $OutRoot $EventId
  New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

  if (Ensure-EventFiles -OutDir $OutDir) {
    Write-Host "[OK]   ${EventId}: ya existe H1+L1 en $OutDir"
    continue
  }

  Write-Host "[MISS] ${EventId}: descargando H1+L1 (duration=$Duration format=$Format) → $OutDir"

  $detailUrl = Get-EventDetailUrl -EventId $EventId
  $detail      = Invoke-RestMethod -Uri $detailUrl -Method GET
$strainFiles = Invoke-RestMethod -Uri $detail.strain_files_url -Method GET

$h1Url = Pick-DetectorUrl -StrainFilesJson $strainFiles -Detector "H1" -Duration $Duration -Format $Format
$l1Url = Pick-DetectorUrl -StrainFilesJson $strainFiles -Detector "L1" -Duration $Duration -Format $Format

  if (-not $h1Url -or -not $l1Url) {
    throw "No encontré URLs H1/L1 para $EventId (duration=$Duration format=$Format). detail_url=$detailUrl"
  }

  $h1Path = Download-File -Url $h1Url -OutDir $OutDir
  $l1Path = Download-File -Url $l1Url -OutDir $OutDir

  Write-Host "  [HASH] $EventId"
  Get-FileHash -Algorithm SHA256 $h1Path | Format-Table -AutoSize
  Get-FileHash -Algorithm SHA256 $l1Path | Format-Table -AutoSize
}

Write-Host "[DONE] Descarga finalizada en $OutRoot"


.download_url } | Sort-Object format,duration | Select-Object -First 1
  if ($cand -and $cand.download_url) { return $cand.download_url }

  return $null
}function Ensure-EventFiles([string]$OutDir) {
  $h1 = Get-ChildItem -Path $OutDir -Filter "*H1*.hdf5" -ErrorAction SilentlyContinue | Select-Object -First 1
  $l1 = Get-ChildItem -Path $OutDir -Filter "*L1*.hdf5" -ErrorAction SilentlyContinue | Select-Object -First 1
  return ($null -ne $h1 -and $null -ne $l1)
}

function Download-File([string]$Url, [string]$OutDir) {
  $fileName = [System.IO.Path]::GetFileName(([uri]$Url).AbsolutePath)
  $outPath = Join-Path $OutDir $fileName

  if (Test-Path $outPath) {
    $len = (Get-Item $outPath).Length
    if ($len -gt 0) {
      Write-Host "  [SKIP] Existe: $outPath ($len bytes)"
      return $outPath
    } else {
      Remove-Item -Force $outPath
    }
  }

  Write-Host "  [GET]  $Url"
  Start-BitsTransfer -Source $Url -Destination $outPath -TransferType Download -ErrorAction Stop
  return $outPath
}

New-Item -ItemType Directory -Force -Path $OutRoot | Out-Null

foreach ($EventId in $Events) {
  $OutDir  = Join-Path $OutRoot $EventId
  New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

  if (Ensure-EventFiles -OutDir $OutDir) {
    Write-Host "[OK]   ${EventId}: ya existe H1+L1 en $OutDir"
    continue
  }

  Write-Host "[MISS] ${EventId}: descargando H1+L1 (duration=$Duration format=$Format) → $OutDir"

  $detailUrl = Get-EventDetailUrl -EventId $EventId
  $detail      = Invoke-RestMethod -Uri $detailUrl -Method GET
$strainFiles = Invoke-RestMethod -Uri $detail.strain_files_url -Method GET

$h1Url = Pick-DetectorUrl -StrainFilesJson $strainFiles -Detector "H1" -Duration $Duration -Format $Format
$l1Url = Pick-DetectorUrl -StrainFilesJson $strainFiles -Detector "L1" -Duration $Duration -Format $Format

  if (-not $h1Url -or -not $l1Url) {
    throw "No encontré URLs H1/L1 para $EventId (duration=$Duration format=$Format). detail_url=$detailUrl"
  }

  $h1Path = Download-File -Url $h1Url -OutDir $OutDir
  $l1Path = Download-File -Url $l1Url -OutDir $OutDir

  Write-Host "  [HASH] $EventId"
  Get-FileHash -Algorithm SHA256 $h1Path | Format-Table -AutoSize
  Get-FileHash -Algorithm SHA256 $l1Path | Format-Table -AutoSize
}

Write-Host "[DONE] Descarga finalizada en $OutRoot"
.Exception.Message)"
  }

  # GWOSC puede devolver 'events' (lista) o 'event' (objeto) o estructuras diferentes.
  $detail = $null

  if ($resp -and $resp.PSObject.Properties.Name -contains "events" -and $resp.events -and $resp.events.Count -gt 0) {
    $ev = $resp.events[0]
    if ($ev.versions -and $ev.versions.Count -gt 0) { $detail = $ev.versions[-1].detail_url }
    if (-not $detail -and $ev.detail_url) { $detail = $ev.detail_url }
  } elseif ($resp -and $resp.PSObject.Properties.Name -contains "event" -and $resp.event) {
    $ev = $resp.event
    if ($ev.versions -and $ev.versions.Count -gt 0) { $detail = $ev.versions[-1].detail_url }
    if (-not $detail -and $ev.detail_url) { $detail = $ev.detail_url }
  } elseif ($resp -and $resp.PSObject.Properties.Name -contains "detail_url") {
    $detail = $resp.detail_url
  }

    if (-not $detail -and $resp -and ($resp.PSObject.Properties.Name -contains "versions") -and $resp.versions -and $resp.versions.Count -gt 0) {
    $detail = $resp.versions[-1].detail_url
  }

  if (-not $detail) {
    # Dump mínimo auditable (sin volcar 10k líneas)
    $keys = if ($resp) { ($resp.PSObject.Properties.Name -join ",") } else { "<null>" }
    throw "No detail_url para ${EventId}. API=$api. Top-level keys=[$keys]"
  }
  return $detail
}function Pick-DetectorUrl($StrainFilesJson, [string]$Detector, [int]$Duration, [string]$Format) {
  # StrainFilesJson suele ser un array (lista) o un objeto con propiedad 'strain_files'
  $files = $null
  if ($StrainFilesJson -is [System.Array]) {
    $files = $StrainFilesJson
  } elseif ($StrainFilesJson -and ($StrainFilesJson.PSObject.Properties.Name -contains "strain_files")) {
    $files = $StrainFilesJson.strain_files
  }

  if (-not $files) { return $null }

  # Preferencia: hdf5 + duration exacta; si no, cualquier hdf5; si no, cualquier formato.
  $cand = $files | Where-Object { $ErrorActionPreference = "Stop"

$RepoRoot = (Get-Location).Path
$OutRoot  = Join-Path $RepoRoot "data\losc"
$Duration = 32
$Format   = "hdf5"

$Events = @(
  "GW150914","GW151226","GW170104","GW170608","GW170729",
  "GW170809","GW170814","GW170818","GW170823","GW190521_030229"
)

function Get-EventDetailUrl([string]$EventId) {
  $api = "https://gwosc.org/api/v2/events/$EventId"
  try {
    $resp = Invoke-RestMethod -Uri $api -Method GET
  } catch {
    throw "GWOSC request failed for ${EventId}. API=$api. $($ErrorActionPreference = "Stop"

$RepoRoot = (Get-Location).Path
$OutRoot  = Join-Path $RepoRoot "data\losc"
$Duration = 32
$Format   = "hdf5"

$Events = @(
  "GW150914","GW151226","GW170104","GW170608","GW170729",
  "GW170809","GW170814","GW170818","GW170823","GW190521_030229"
)

function Get-EventDetailUrl([string]$EventId) {
  $api = "https://gwosc.org/api/v2/events/$EventId"
  $resp = Invoke-RestMethod -Uri $api -Method GET
  $detail = $resp.events[0].versions[-1].detail_url
  if (-not $detail) { throw "No detail_url para $EventId (API=$api)" }
  return $detail
}

function Pick-DetectorUrl($DetailJson, [string]$Detector, [int]$Duration, [string]$Format) {
  $strain = $DetailJson.strain | Where-Object { $_.detector -eq $Detector }
  if (-not $strain) { return $null }
  $file = $strain.files | Where-Object { $_.format -eq $Format -and $_.duration -eq $Duration } | Select-Object -First 1
  if (-not $file) { return $null }
  return $file.download_url
}

function Ensure-EventFiles([string]$OutDir) {
  $h1 = Get-ChildItem -Path $OutDir -Filter "*H1*.hdf5" -ErrorAction SilentlyContinue | Select-Object -First 1
  $l1 = Get-ChildItem -Path $OutDir -Filter "*L1*.hdf5" -ErrorAction SilentlyContinue | Select-Object -First 1
  return ($null -ne $h1 -and $null -ne $l1)
}

function Download-File([string]$Url, [string]$OutDir) {
  $fileName = [System.IO.Path]::GetFileName(([uri]$Url).AbsolutePath)
  $outPath = Join-Path $OutDir $fileName

  if (Test-Path $outPath) {
    $len = (Get-Item $outPath).Length
    if ($len -gt 0) {
      Write-Host "  [SKIP] Existe: $outPath ($len bytes)"
      return $outPath
    } else {
      Remove-Item -Force $outPath
    }
  }

  Write-Host "  [GET]  $Url"
  Start-BitsTransfer -Source $Url -Destination $outPath -TransferType Download -ErrorAction Stop
  return $outPath
}

New-Item -ItemType Directory -Force -Path $OutRoot | Out-Null

foreach ($EventId in $Events) {
  $OutDir  = Join-Path $OutRoot $EventId
  New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

  if (Ensure-EventFiles -OutDir $OutDir) {
    Write-Host "[OK]   ${EventId}: ya existe H1+L1 en $OutDir"
    continue
  }

  Write-Host "[MISS] ${EventId}: descargando H1+L1 (duration=$Duration format=$Format) → $OutDir"

  $detailUrl = Get-EventDetailUrl -EventId $EventId
  $detail      = Invoke-RestMethod -Uri $detailUrl -Method GET
$strainFiles = Invoke-RestMethod -Uri $detail.strain_files_url -Method GET

$h1Url = Pick-DetectorUrl -StrainFilesJson $strainFiles -Detector "H1" -Duration $Duration -Format $Format
$l1Url = Pick-DetectorUrl -StrainFilesJson $strainFiles -Detector "L1" -Duration $Duration -Format $Format

  if (-not $h1Url -or -not $l1Url) {
    throw "No encontré URLs H1/L1 para $EventId (duration=$Duration format=$Format). detail_url=$detailUrl"
  }

  $h1Path = Download-File -Url $h1Url -OutDir $OutDir
  $l1Path = Download-File -Url $l1Url -OutDir $OutDir

  Write-Host "  [HASH] $EventId"
  Get-FileHash -Algorithm SHA256 $h1Path | Format-Table -AutoSize
  Get-FileHash -Algorithm SHA256 $l1Path | Format-Table -AutoSize
}

Write-Host "[DONE] Descarga finalizada en $OutRoot"
.Exception.Message)"
  }

  # GWOSC puede devolver 'events' (lista) o 'event' (objeto) o estructuras diferentes.
  $detail = $null

  if ($resp -and $resp.PSObject.Properties.Name -contains "events" -and $resp.events -and $resp.events.Count -gt 0) {
    $ev = $resp.events[0]
    if ($ev.versions -and $ev.versions.Count -gt 0) { $detail = $ev.versions[-1].detail_url }
    if (-not $detail -and $ev.detail_url) { $detail = $ev.detail_url }
  } elseif ($resp -and $resp.PSObject.Properties.Name -contains "event" -and $resp.event) {
    $ev = $resp.event
    if ($ev.versions -and $ev.versions.Count -gt 0) { $detail = $ev.versions[-1].detail_url }
    if (-not $detail -and $ev.detail_url) { $detail = $ev.detail_url }
  } elseif ($resp -and $resp.PSObject.Properties.Name -contains "detail_url") {
    $detail = $resp.detail_url
  }

    if (-not $detail -and $resp -and ($resp.PSObject.Properties.Name -contains "versions") -and $resp.versions -and $resp.versions.Count -gt 0) {
    $detail = $resp.versions[-1].detail_url
  }

  if (-not $detail) {
    # Dump mínimo auditable (sin volcar 10k líneas)
    $keys = if ($resp) { ($resp.PSObject.Properties.Name -join ",") } else { "<null>" }
    throw "No detail_url para ${EventId}. API=$api. Top-level keys=[$keys]"
  }
  return $detail
}function Pick-DetectorUrl($DetailJson, [string]$Detector, [int]$Duration, [string]$Format) {
  $strain = $DetailJson.strain | Where-Object { $_.detector -eq $Detector }
  if (-not $strain) { return $null }
  $file = $strain.files | Where-Object { $_.format -eq $Format -and $_.duration -eq $Duration } | Select-Object -First 1
  if (-not $file) { return $null }
  return $file.download_url
}

function Ensure-EventFiles([string]$OutDir) {
  $h1 = Get-ChildItem -Path $OutDir -Filter "*H1*.hdf5" -ErrorAction SilentlyContinue | Select-Object -First 1
  $l1 = Get-ChildItem -Path $OutDir -Filter "*L1*.hdf5" -ErrorAction SilentlyContinue | Select-Object -First 1
  return ($null -ne $h1 -and $null -ne $l1)
}

function Download-File([string]$Url, [string]$OutDir) {
  $fileName = [System.IO.Path]::GetFileName(([uri]$Url).AbsolutePath)
  $outPath = Join-Path $OutDir $fileName

  if (Test-Path $outPath) {
    $len = (Get-Item $outPath).Length
    if ($len -gt 0) {
      Write-Host "  [SKIP] Existe: $outPath ($len bytes)"
      return $outPath
    } else {
      Remove-Item -Force $outPath
    }
  }

  Write-Host "  [GET]  $Url"
  Start-BitsTransfer -Source $Url -Destination $outPath -TransferType Download -ErrorAction Stop
  return $outPath
}

New-Item -ItemType Directory -Force -Path $OutRoot | Out-Null

foreach ($EventId in $Events) {
  $OutDir  = Join-Path $OutRoot $EventId
  New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

  if (Ensure-EventFiles -OutDir $OutDir) {
    Write-Host "[OK]   ${EventId}: ya existe H1+L1 en $OutDir"
    continue
  }

  Write-Host "[MISS] ${EventId}: descargando H1+L1 (duration=$Duration format=$Format) → $OutDir"

  $detailUrl = Get-EventDetailUrl -EventId $EventId
  $detail      = Invoke-RestMethod -Uri $detailUrl -Method GET
$strainFiles = Invoke-RestMethod -Uri $detail.strain_files_url -Method GET

$h1Url = Pick-DetectorUrl -StrainFilesJson $strainFiles -Detector "H1" -Duration $Duration -Format $Format
$l1Url = Pick-DetectorUrl -StrainFilesJson $strainFiles -Detector "L1" -Duration $Duration -Format $Format

  if (-not $h1Url -or -not $l1Url) {
    throw "No encontré URLs H1/L1 para $EventId (duration=$Duration format=$Format). detail_url=$detailUrl"
  }

  $h1Path = Download-File -Url $h1Url -OutDir $OutDir
  $l1Path = Download-File -Url $l1Url -OutDir $OutDir

  Write-Host "  [HASH] $EventId"
  Get-FileHash -Algorithm SHA256 $h1Path | Format-Table -AutoSize
  Get-FileHash -Algorithm SHA256 $l1Path | Format-Table -AutoSize
}

Write-Host "[DONE] Descarga finalizada en $OutRoot"


.detector -eq $Detector -and $ErrorActionPreference = "Stop"

$RepoRoot = (Get-Location).Path
$OutRoot  = Join-Path $RepoRoot "data\losc"
$Duration = 32
$Format   = "hdf5"

$Events = @(
  "GW150914","GW151226","GW170104","GW170608","GW170729",
  "GW170809","GW170814","GW170818","GW170823","GW190521_030229"
)

function Get-EventDetailUrl([string]$EventId) {
  $api = "https://gwosc.org/api/v2/events/$EventId"
  try {
    $resp = Invoke-RestMethod -Uri $api -Method GET
  } catch {
    throw "GWOSC request failed for ${EventId}. API=$api. $($ErrorActionPreference = "Stop"

$RepoRoot = (Get-Location).Path
$OutRoot  = Join-Path $RepoRoot "data\losc"
$Duration = 32
$Format   = "hdf5"

$Events = @(
  "GW150914","GW151226","GW170104","GW170608","GW170729",
  "GW170809","GW170814","GW170818","GW170823","GW190521_030229"
)

function Get-EventDetailUrl([string]$EventId) {
  $api = "https://gwosc.org/api/v2/events/$EventId"
  $resp = Invoke-RestMethod -Uri $api -Method GET
  $detail = $resp.events[0].versions[-1].detail_url
  if (-not $detail) { throw "No detail_url para $EventId (API=$api)" }
  return $detail
}

function Pick-DetectorUrl($DetailJson, [string]$Detector, [int]$Duration, [string]$Format) {
  $strain = $DetailJson.strain | Where-Object { $_.detector -eq $Detector }
  if (-not $strain) { return $null }
  $file = $strain.files | Where-Object { $_.format -eq $Format -and $_.duration -eq $Duration } | Select-Object -First 1
  if (-not $file) { return $null }
  return $file.download_url
}

function Ensure-EventFiles([string]$OutDir) {
  $h1 = Get-ChildItem -Path $OutDir -Filter "*H1*.hdf5" -ErrorAction SilentlyContinue | Select-Object -First 1
  $l1 = Get-ChildItem -Path $OutDir -Filter "*L1*.hdf5" -ErrorAction SilentlyContinue | Select-Object -First 1
  return ($null -ne $h1 -and $null -ne $l1)
}

function Download-File([string]$Url, [string]$OutDir) {
  $fileName = [System.IO.Path]::GetFileName(([uri]$Url).AbsolutePath)
  $outPath = Join-Path $OutDir $fileName

  if (Test-Path $outPath) {
    $len = (Get-Item $outPath).Length
    if ($len -gt 0) {
      Write-Host "  [SKIP] Existe: $outPath ($len bytes)"
      return $outPath
    } else {
      Remove-Item -Force $outPath
    }
  }

  Write-Host "  [GET]  $Url"
  Start-BitsTransfer -Source $Url -Destination $outPath -TransferType Download -ErrorAction Stop
  return $outPath
}

New-Item -ItemType Directory -Force -Path $OutRoot | Out-Null

foreach ($EventId in $Events) {
  $OutDir  = Join-Path $OutRoot $EventId
  New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

  if (Ensure-EventFiles -OutDir $OutDir) {
    Write-Host "[OK]   ${EventId}: ya existe H1+L1 en $OutDir"
    continue
  }

  Write-Host "[MISS] ${EventId}: descargando H1+L1 (duration=$Duration format=$Format) → $OutDir"

  $detailUrl = Get-EventDetailUrl -EventId $EventId
  $detail      = Invoke-RestMethod -Uri $detailUrl -Method GET
$strainFiles = Invoke-RestMethod -Uri $detail.strain_files_url -Method GET

$h1Url = Pick-DetectorUrl -StrainFilesJson $strainFiles -Detector "H1" -Duration $Duration -Format $Format
$l1Url = Pick-DetectorUrl -StrainFilesJson $strainFiles -Detector "L1" -Duration $Duration -Format $Format

  if (-not $h1Url -or -not $l1Url) {
    throw "No encontré URLs H1/L1 para $EventId (duration=$Duration format=$Format). detail_url=$detailUrl"
  }

  $h1Path = Download-File -Url $h1Url -OutDir $OutDir
  $l1Path = Download-File -Url $l1Url -OutDir $OutDir

  Write-Host "  [HASH] $EventId"
  Get-FileHash -Algorithm SHA256 $h1Path | Format-Table -AutoSize
  Get-FileHash -Algorithm SHA256 $l1Path | Format-Table -AutoSize
}

Write-Host "[DONE] Descarga finalizada en $OutRoot"
.Exception.Message)"
  }

  # GWOSC puede devolver 'events' (lista) o 'event' (objeto) o estructuras diferentes.
  $detail = $null

  if ($resp -and $resp.PSObject.Properties.Name -contains "events" -and $resp.events -and $resp.events.Count -gt 0) {
    $ev = $resp.events[0]
    if ($ev.versions -and $ev.versions.Count -gt 0) { $detail = $ev.versions[-1].detail_url }
    if (-not $detail -and $ev.detail_url) { $detail = $ev.detail_url }
  } elseif ($resp -and $resp.PSObject.Properties.Name -contains "event" -and $resp.event) {
    $ev = $resp.event
    if ($ev.versions -and $ev.versions.Count -gt 0) { $detail = $ev.versions[-1].detail_url }
    if (-not $detail -and $ev.detail_url) { $detail = $ev.detail_url }
  } elseif ($resp -and $resp.PSObject.Properties.Name -contains "detail_url") {
    $detail = $resp.detail_url
  }

    if (-not $detail -and $resp -and ($resp.PSObject.Properties.Name -contains "versions") -and $resp.versions -and $resp.versions.Count -gt 0) {
    $detail = $resp.versions[-1].detail_url
  }

  if (-not $detail) {
    # Dump mínimo auditable (sin volcar 10k líneas)
    $keys = if ($resp) { ($resp.PSObject.Properties.Name -join ",") } else { "<null>" }
    throw "No detail_url para ${EventId}. API=$api. Top-level keys=[$keys]"
  }
  return $detail
}function Pick-DetectorUrl($DetailJson, [string]$Detector, [int]$Duration, [string]$Format) {
  $strain = $DetailJson.strain | Where-Object { $_.detector -eq $Detector }
  if (-not $strain) { return $null }
  $file = $strain.files | Where-Object { $_.format -eq $Format -and $_.duration -eq $Duration } | Select-Object -First 1
  if (-not $file) { return $null }
  return $file.download_url
}

function Ensure-EventFiles([string]$OutDir) {
  $h1 = Get-ChildItem -Path $OutDir -Filter "*H1*.hdf5" -ErrorAction SilentlyContinue | Select-Object -First 1
  $l1 = Get-ChildItem -Path $OutDir -Filter "*L1*.hdf5" -ErrorAction SilentlyContinue | Select-Object -First 1
  return ($null -ne $h1 -and $null -ne $l1)
}

function Download-File([string]$Url, [string]$OutDir) {
  $fileName = [System.IO.Path]::GetFileName(([uri]$Url).AbsolutePath)
  $outPath = Join-Path $OutDir $fileName

  if (Test-Path $outPath) {
    $len = (Get-Item $outPath).Length
    if ($len -gt 0) {
      Write-Host "  [SKIP] Existe: $outPath ($len bytes)"
      return $outPath
    } else {
      Remove-Item -Force $outPath
    }
  }

  Write-Host "  [GET]  $Url"
  Start-BitsTransfer -Source $Url -Destination $outPath -TransferType Download -ErrorAction Stop
  return $outPath
}

New-Item -ItemType Directory -Force -Path $OutRoot | Out-Null

foreach ($EventId in $Events) {
  $OutDir  = Join-Path $OutRoot $EventId
  New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

  if (Ensure-EventFiles -OutDir $OutDir) {
    Write-Host "[OK]   ${EventId}: ya existe H1+L1 en $OutDir"
    continue
  }

  Write-Host "[MISS] ${EventId}: descargando H1+L1 (duration=$Duration format=$Format) → $OutDir"

  $detailUrl = Get-EventDetailUrl -EventId $EventId
  $detail      = Invoke-RestMethod -Uri $detailUrl -Method GET
$strainFiles = Invoke-RestMethod -Uri $detail.strain_files_url -Method GET

$h1Url = Pick-DetectorUrl -StrainFilesJson $strainFiles -Detector "H1" -Duration $Duration -Format $Format
$l1Url = Pick-DetectorUrl -StrainFilesJson $strainFiles -Detector "L1" -Duration $Duration -Format $Format

  if (-not $h1Url -or -not $l1Url) {
    throw "No encontré URLs H1/L1 para $EventId (duration=$Duration format=$Format). detail_url=$detailUrl"
  }

  $h1Path = Download-File -Url $h1Url -OutDir $OutDir
  $l1Path = Download-File -Url $l1Url -OutDir $OutDir

  Write-Host "  [HASH] $EventId"
  Get-FileHash -Algorithm SHA256 $h1Path | Format-Table -AutoSize
  Get-FileHash -Algorithm SHA256 $l1Path | Format-Table -AutoSize
}

Write-Host "[DONE] Descarga finalizada en $OutRoot"


.format -eq $Format -and $ErrorActionPreference = "Stop"

$RepoRoot = (Get-Location).Path
$OutRoot  = Join-Path $RepoRoot "data\losc"
$Duration = 32
$Format   = "hdf5"

$Events = @(
  "GW150914","GW151226","GW170104","GW170608","GW170729",
  "GW170809","GW170814","GW170818","GW170823","GW190521_030229"
)

function Get-EventDetailUrl([string]$EventId) {
  $api = "https://gwosc.org/api/v2/events/$EventId"
  try {
    $resp = Invoke-RestMethod -Uri $api -Method GET
  } catch {
    throw "GWOSC request failed for ${EventId}. API=$api. $($ErrorActionPreference = "Stop"

$RepoRoot = (Get-Location).Path
$OutRoot  = Join-Path $RepoRoot "data\losc"
$Duration = 32
$Format   = "hdf5"

$Events = @(
  "GW150914","GW151226","GW170104","GW170608","GW170729",
  "GW170809","GW170814","GW170818","GW170823","GW190521_030229"
)

function Get-EventDetailUrl([string]$EventId) {
  $api = "https://gwosc.org/api/v2/events/$EventId"
  $resp = Invoke-RestMethod -Uri $api -Method GET
  $detail = $resp.events[0].versions[-1].detail_url
  if (-not $detail) { throw "No detail_url para $EventId (API=$api)" }
  return $detail
}

function Pick-DetectorUrl($DetailJson, [string]$Detector, [int]$Duration, [string]$Format) {
  $strain = $DetailJson.strain | Where-Object { $_.detector -eq $Detector }
  if (-not $strain) { return $null }
  $file = $strain.files | Where-Object { $_.format -eq $Format -and $_.duration -eq $Duration } | Select-Object -First 1
  if (-not $file) { return $null }
  return $file.download_url
}

function Ensure-EventFiles([string]$OutDir) {
  $h1 = Get-ChildItem -Path $OutDir -Filter "*H1*.hdf5" -ErrorAction SilentlyContinue | Select-Object -First 1
  $l1 = Get-ChildItem -Path $OutDir -Filter "*L1*.hdf5" -ErrorAction SilentlyContinue | Select-Object -First 1
  return ($null -ne $h1 -and $null -ne $l1)
}

function Download-File([string]$Url, [string]$OutDir) {
  $fileName = [System.IO.Path]::GetFileName(([uri]$Url).AbsolutePath)
  $outPath = Join-Path $OutDir $fileName

  if (Test-Path $outPath) {
    $len = (Get-Item $outPath).Length
    if ($len -gt 0) {
      Write-Host "  [SKIP] Existe: $outPath ($len bytes)"
      return $outPath
    } else {
      Remove-Item -Force $outPath
    }
  }

  Write-Host "  [GET]  $Url"
  Start-BitsTransfer -Source $Url -Destination $outPath -TransferType Download -ErrorAction Stop
  return $outPath
}

New-Item -ItemType Directory -Force -Path $OutRoot | Out-Null

foreach ($EventId in $Events) {
  $OutDir  = Join-Path $OutRoot $EventId
  New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

  if (Ensure-EventFiles -OutDir $OutDir) {
    Write-Host "[OK]   ${EventId}: ya existe H1+L1 en $OutDir"
    continue
  }

  Write-Host "[MISS] ${EventId}: descargando H1+L1 (duration=$Duration format=$Format) → $OutDir"

  $detailUrl = Get-EventDetailUrl -EventId $EventId
  $detail      = Invoke-RestMethod -Uri $detailUrl -Method GET
$strainFiles = Invoke-RestMethod -Uri $detail.strain_files_url -Method GET

$h1Url = Pick-DetectorUrl -StrainFilesJson $strainFiles -Detector "H1" -Duration $Duration -Format $Format
$l1Url = Pick-DetectorUrl -StrainFilesJson $strainFiles -Detector "L1" -Duration $Duration -Format $Format

  if (-not $h1Url -or -not $l1Url) {
    throw "No encontré URLs H1/L1 para $EventId (duration=$Duration format=$Format). detail_url=$detailUrl"
  }

  $h1Path = Download-File -Url $h1Url -OutDir $OutDir
  $l1Path = Download-File -Url $l1Url -OutDir $OutDir

  Write-Host "  [HASH] $EventId"
  Get-FileHash -Algorithm SHA256 $h1Path | Format-Table -AutoSize
  Get-FileHash -Algorithm SHA256 $l1Path | Format-Table -AutoSize
}

Write-Host "[DONE] Descarga finalizada en $OutRoot"
.Exception.Message)"
  }

  # GWOSC puede devolver 'events' (lista) o 'event' (objeto) o estructuras diferentes.
  $detail = $null

  if ($resp -and $resp.PSObject.Properties.Name -contains "events" -and $resp.events -and $resp.events.Count -gt 0) {
    $ev = $resp.events[0]
    if ($ev.versions -and $ev.versions.Count -gt 0) { $detail = $ev.versions[-1].detail_url }
    if (-not $detail -and $ev.detail_url) { $detail = $ev.detail_url }
  } elseif ($resp -and $resp.PSObject.Properties.Name -contains "event" -and $resp.event) {
    $ev = $resp.event
    if ($ev.versions -and $ev.versions.Count -gt 0) { $detail = $ev.versions[-1].detail_url }
    if (-not $detail -and $ev.detail_url) { $detail = $ev.detail_url }
  } elseif ($resp -and $resp.PSObject.Properties.Name -contains "detail_url") {
    $detail = $resp.detail_url
  }

    if (-not $detail -and $resp -and ($resp.PSObject.Properties.Name -contains "versions") -and $resp.versions -and $resp.versions.Count -gt 0) {
    $detail = $resp.versions[-1].detail_url
  }

  if (-not $detail) {
    # Dump mínimo auditable (sin volcar 10k líneas)
    $keys = if ($resp) { ($resp.PSObject.Properties.Name -join ",") } else { "<null>" }
    throw "No detail_url para ${EventId}. API=$api. Top-level keys=[$keys]"
  }
  return $detail
}function Pick-DetectorUrl($DetailJson, [string]$Detector, [int]$Duration, [string]$Format) {
  $strain = $DetailJson.strain | Where-Object { $_.detector -eq $Detector }
  if (-not $strain) { return $null }
  $file = $strain.files | Where-Object { $_.format -eq $Format -and $_.duration -eq $Duration } | Select-Object -First 1
  if (-not $file) { return $null }
  return $file.download_url
}

function Ensure-EventFiles([string]$OutDir) {
  $h1 = Get-ChildItem -Path $OutDir -Filter "*H1*.hdf5" -ErrorAction SilentlyContinue | Select-Object -First 1
  $l1 = Get-ChildItem -Path $OutDir -Filter "*L1*.hdf5" -ErrorAction SilentlyContinue | Select-Object -First 1
  return ($null -ne $h1 -and $null -ne $l1)
}

function Download-File([string]$Url, [string]$OutDir) {
  $fileName = [System.IO.Path]::GetFileName(([uri]$Url).AbsolutePath)
  $outPath = Join-Path $OutDir $fileName

  if (Test-Path $outPath) {
    $len = (Get-Item $outPath).Length
    if ($len -gt 0) {
      Write-Host "  [SKIP] Existe: $outPath ($len bytes)"
      return $outPath
    } else {
      Remove-Item -Force $outPath
    }
  }

  Write-Host "  [GET]  $Url"
  Start-BitsTransfer -Source $Url -Destination $outPath -TransferType Download -ErrorAction Stop
  return $outPath
}

New-Item -ItemType Directory -Force -Path $OutRoot | Out-Null

foreach ($EventId in $Events) {
  $OutDir  = Join-Path $OutRoot $EventId
  New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

  if (Ensure-EventFiles -OutDir $OutDir) {
    Write-Host "[OK]   ${EventId}: ya existe H1+L1 en $OutDir"
    continue
  }

  Write-Host "[MISS] ${EventId}: descargando H1+L1 (duration=$Duration format=$Format) → $OutDir"

  $detailUrl = Get-EventDetailUrl -EventId $EventId
  $detail      = Invoke-RestMethod -Uri $detailUrl -Method GET
$strainFiles = Invoke-RestMethod -Uri $detail.strain_files_url -Method GET

$h1Url = Pick-DetectorUrl -StrainFilesJson $strainFiles -Detector "H1" -Duration $Duration -Format $Format
$l1Url = Pick-DetectorUrl -StrainFilesJson $strainFiles -Detector "L1" -Duration $Duration -Format $Format

  if (-not $h1Url -or -not $l1Url) {
    throw "No encontré URLs H1/L1 para $EventId (duration=$Duration format=$Format). detail_url=$detailUrl"
  }

  $h1Path = Download-File -Url $h1Url -OutDir $OutDir
  $l1Path = Download-File -Url $l1Url -OutDir $OutDir

  Write-Host "  [HASH] $EventId"
  Get-FileHash -Algorithm SHA256 $h1Path | Format-Table -AutoSize
  Get-FileHash -Algorithm SHA256 $l1Path | Format-Table -AutoSize
}

Write-Host "[DONE] Descarga finalizada en $OutRoot"


.duration -eq $Duration } | Select-Object -First 1
  if ($cand -and $cand.download_url) { return $cand.download_url }

  $cand = $files | Where-Object { $ErrorActionPreference = "Stop"

$RepoRoot = (Get-Location).Path
$OutRoot  = Join-Path $RepoRoot "data\losc"
$Duration = 32
$Format   = "hdf5"

$Events = @(
  "GW150914","GW151226","GW170104","GW170608","GW170729",
  "GW170809","GW170814","GW170818","GW170823","GW190521_030229"
)

function Get-EventDetailUrl([string]$EventId) {
  $api = "https://gwosc.org/api/v2/events/$EventId"
  try {
    $resp = Invoke-RestMethod -Uri $api -Method GET
  } catch {
    throw "GWOSC request failed for ${EventId}. API=$api. $($ErrorActionPreference = "Stop"

$RepoRoot = (Get-Location).Path
$OutRoot  = Join-Path $RepoRoot "data\losc"
$Duration = 32
$Format   = "hdf5"

$Events = @(
  "GW150914","GW151226","GW170104","GW170608","GW170729",
  "GW170809","GW170814","GW170818","GW170823","GW190521_030229"
)

function Get-EventDetailUrl([string]$EventId) {
  $api = "https://gwosc.org/api/v2/events/$EventId"
  $resp = Invoke-RestMethod -Uri $api -Method GET
  $detail = $resp.events[0].versions[-1].detail_url
  if (-not $detail) { throw "No detail_url para $EventId (API=$api)" }
  return $detail
}

function Pick-DetectorUrl($DetailJson, [string]$Detector, [int]$Duration, [string]$Format) {
  $strain = $DetailJson.strain | Where-Object { $_.detector -eq $Detector }
  if (-not $strain) { return $null }
  $file = $strain.files | Where-Object { $_.format -eq $Format -and $_.duration -eq $Duration } | Select-Object -First 1
  if (-not $file) { return $null }
  return $file.download_url
}

function Ensure-EventFiles([string]$OutDir) {
  $h1 = Get-ChildItem -Path $OutDir -Filter "*H1*.hdf5" -ErrorAction SilentlyContinue | Select-Object -First 1
  $l1 = Get-ChildItem -Path $OutDir -Filter "*L1*.hdf5" -ErrorAction SilentlyContinue | Select-Object -First 1
  return ($null -ne $h1 -and $null -ne $l1)
}

function Download-File([string]$Url, [string]$OutDir) {
  $fileName = [System.IO.Path]::GetFileName(([uri]$Url).AbsolutePath)
  $outPath = Join-Path $OutDir $fileName

  if (Test-Path $outPath) {
    $len = (Get-Item $outPath).Length
    if ($len -gt 0) {
      Write-Host "  [SKIP] Existe: $outPath ($len bytes)"
      return $outPath
    } else {
      Remove-Item -Force $outPath
    }
  }

  Write-Host "  [GET]  $Url"
  Start-BitsTransfer -Source $Url -Destination $outPath -TransferType Download -ErrorAction Stop
  return $outPath
}

New-Item -ItemType Directory -Force -Path $OutRoot | Out-Null

foreach ($EventId in $Events) {
  $OutDir  = Join-Path $OutRoot $EventId
  New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

  if (Ensure-EventFiles -OutDir $OutDir) {
    Write-Host "[OK]   ${EventId}: ya existe H1+L1 en $OutDir"
    continue
  }

  Write-Host "[MISS] ${EventId}: descargando H1+L1 (duration=$Duration format=$Format) → $OutDir"

  $detailUrl = Get-EventDetailUrl -EventId $EventId
  $detail      = Invoke-RestMethod -Uri $detailUrl -Method GET
$strainFiles = Invoke-RestMethod -Uri $detail.strain_files_url -Method GET

$h1Url = Pick-DetectorUrl -StrainFilesJson $strainFiles -Detector "H1" -Duration $Duration -Format $Format
$l1Url = Pick-DetectorUrl -StrainFilesJson $strainFiles -Detector "L1" -Duration $Duration -Format $Format

  if (-not $h1Url -or -not $l1Url) {
    throw "No encontré URLs H1/L1 para $EventId (duration=$Duration format=$Format). detail_url=$detailUrl"
  }

  $h1Path = Download-File -Url $h1Url -OutDir $OutDir
  $l1Path = Download-File -Url $l1Url -OutDir $OutDir

  Write-Host "  [HASH] $EventId"
  Get-FileHash -Algorithm SHA256 $h1Path | Format-Table -AutoSize
  Get-FileHash -Algorithm SHA256 $l1Path | Format-Table -AutoSize
}

Write-Host "[DONE] Descarga finalizada en $OutRoot"
.Exception.Message)"
  }

  # GWOSC puede devolver 'events' (lista) o 'event' (objeto) o estructuras diferentes.
  $detail = $null

  if ($resp -and $resp.PSObject.Properties.Name -contains "events" -and $resp.events -and $resp.events.Count -gt 0) {
    $ev = $resp.events[0]
    if ($ev.versions -and $ev.versions.Count -gt 0) { $detail = $ev.versions[-1].detail_url }
    if (-not $detail -and $ev.detail_url) { $detail = $ev.detail_url }
  } elseif ($resp -and $resp.PSObject.Properties.Name -contains "event" -and $resp.event) {
    $ev = $resp.event
    if ($ev.versions -and $ev.versions.Count -gt 0) { $detail = $ev.versions[-1].detail_url }
    if (-not $detail -and $ev.detail_url) { $detail = $ev.detail_url }
  } elseif ($resp -and $resp.PSObject.Properties.Name -contains "detail_url") {
    $detail = $resp.detail_url
  }

    if (-not $detail -and $resp -and ($resp.PSObject.Properties.Name -contains "versions") -and $resp.versions -and $resp.versions.Count -gt 0) {
    $detail = $resp.versions[-1].detail_url
  }

  if (-not $detail) {
    # Dump mínimo auditable (sin volcar 10k líneas)
    $keys = if ($resp) { ($resp.PSObject.Properties.Name -join ",") } else { "<null>" }
    throw "No detail_url para ${EventId}. API=$api. Top-level keys=[$keys]"
  }
  return $detail
}function Pick-DetectorUrl($DetailJson, [string]$Detector, [int]$Duration, [string]$Format) {
  $strain = $DetailJson.strain | Where-Object { $_.detector -eq $Detector }
  if (-not $strain) { return $null }
  $file = $strain.files | Where-Object { $_.format -eq $Format -and $_.duration -eq $Duration } | Select-Object -First 1
  if (-not $file) { return $null }
  return $file.download_url
}

function Ensure-EventFiles([string]$OutDir) {
  $h1 = Get-ChildItem -Path $OutDir -Filter "*H1*.hdf5" -ErrorAction SilentlyContinue | Select-Object -First 1
  $l1 = Get-ChildItem -Path $OutDir -Filter "*L1*.hdf5" -ErrorAction SilentlyContinue | Select-Object -First 1
  return ($null -ne $h1 -and $null -ne $l1)
}

function Download-File([string]$Url, [string]$OutDir) {
  $fileName = [System.IO.Path]::GetFileName(([uri]$Url).AbsolutePath)
  $outPath = Join-Path $OutDir $fileName

  if (Test-Path $outPath) {
    $len = (Get-Item $outPath).Length
    if ($len -gt 0) {
      Write-Host "  [SKIP] Existe: $outPath ($len bytes)"
      return $outPath
    } else {
      Remove-Item -Force $outPath
    }
  }

  Write-Host "  [GET]  $Url"
  Start-BitsTransfer -Source $Url -Destination $outPath -TransferType Download -ErrorAction Stop
  return $outPath
}

New-Item -ItemType Directory -Force -Path $OutRoot | Out-Null

foreach ($EventId in $Events) {
  $OutDir  = Join-Path $OutRoot $EventId
  New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

  if (Ensure-EventFiles -OutDir $OutDir) {
    Write-Host "[OK]   ${EventId}: ya existe H1+L1 en $OutDir"
    continue
  }

  Write-Host "[MISS] ${EventId}: descargando H1+L1 (duration=$Duration format=$Format) → $OutDir"

  $detailUrl = Get-EventDetailUrl -EventId $EventId
  $detail      = Invoke-RestMethod -Uri $detailUrl -Method GET
$strainFiles = Invoke-RestMethod -Uri $detail.strain_files_url -Method GET

$h1Url = Pick-DetectorUrl -StrainFilesJson $strainFiles -Detector "H1" -Duration $Duration -Format $Format
$l1Url = Pick-DetectorUrl -StrainFilesJson $strainFiles -Detector "L1" -Duration $Duration -Format $Format

  if (-not $h1Url -or -not $l1Url) {
    throw "No encontré URLs H1/L1 para $EventId (duration=$Duration format=$Format). detail_url=$detailUrl"
  }

  $h1Path = Download-File -Url $h1Url -OutDir $OutDir
  $l1Path = Download-File -Url $l1Url -OutDir $OutDir

  Write-Host "  [HASH] $EventId"
  Get-FileHash -Algorithm SHA256 $h1Path | Format-Table -AutoSize
  Get-FileHash -Algorithm SHA256 $l1Path | Format-Table -AutoSize
}

Write-Host "[DONE] Descarga finalizada en $OutRoot"


.detector -eq $Detector -and $ErrorActionPreference = "Stop"

$RepoRoot = (Get-Location).Path
$OutRoot  = Join-Path $RepoRoot "data\losc"
$Duration = 32
$Format   = "hdf5"

$Events = @(
  "GW150914","GW151226","GW170104","GW170608","GW170729",
  "GW170809","GW170814","GW170818","GW170823","GW190521_030229"
)

function Get-EventDetailUrl([string]$EventId) {
  $api = "https://gwosc.org/api/v2/events/$EventId"
  try {
    $resp = Invoke-RestMethod -Uri $api -Method GET
  } catch {
    throw "GWOSC request failed for ${EventId}. API=$api. $($ErrorActionPreference = "Stop"

$RepoRoot = (Get-Location).Path
$OutRoot  = Join-Path $RepoRoot "data\losc"
$Duration = 32
$Format   = "hdf5"

$Events = @(
  "GW150914","GW151226","GW170104","GW170608","GW170729",
  "GW170809","GW170814","GW170818","GW170823","GW190521_030229"
)

function Get-EventDetailUrl([string]$EventId) {
  $api = "https://gwosc.org/api/v2/events/$EventId"
  $resp = Invoke-RestMethod -Uri $api -Method GET
  $detail = $resp.events[0].versions[-1].detail_url
  if (-not $detail) { throw "No detail_url para $EventId (API=$api)" }
  return $detail
}

function Pick-DetectorUrl($DetailJson, [string]$Detector, [int]$Duration, [string]$Format) {
  $strain = $DetailJson.strain | Where-Object { $_.detector -eq $Detector }
  if (-not $strain) { return $null }
  $file = $strain.files | Where-Object { $_.format -eq $Format -and $_.duration -eq $Duration } | Select-Object -First 1
  if (-not $file) { return $null }
  return $file.download_url
}

function Ensure-EventFiles([string]$OutDir) {
  $h1 = Get-ChildItem -Path $OutDir -Filter "*H1*.hdf5" -ErrorAction SilentlyContinue | Select-Object -First 1
  $l1 = Get-ChildItem -Path $OutDir -Filter "*L1*.hdf5" -ErrorAction SilentlyContinue | Select-Object -First 1
  return ($null -ne $h1 -and $null -ne $l1)
}

function Download-File([string]$Url, [string]$OutDir) {
  $fileName = [System.IO.Path]::GetFileName(([uri]$Url).AbsolutePath)
  $outPath = Join-Path $OutDir $fileName

  if (Test-Path $outPath) {
    $len = (Get-Item $outPath).Length
    if ($len -gt 0) {
      Write-Host "  [SKIP] Existe: $outPath ($len bytes)"
      return $outPath
    } else {
      Remove-Item -Force $outPath
    }
  }

  Write-Host "  [GET]  $Url"
  Start-BitsTransfer -Source $Url -Destination $outPath -TransferType Download -ErrorAction Stop
  return $outPath
}

New-Item -ItemType Directory -Force -Path $OutRoot | Out-Null

foreach ($EventId in $Events) {
  $OutDir  = Join-Path $OutRoot $EventId
  New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

  if (Ensure-EventFiles -OutDir $OutDir) {
    Write-Host "[OK]   ${EventId}: ya existe H1+L1 en $OutDir"
    continue
  }

  Write-Host "[MISS] ${EventId}: descargando H1+L1 (duration=$Duration format=$Format) → $OutDir"

  $detailUrl = Get-EventDetailUrl -EventId $EventId
  $detail      = Invoke-RestMethod -Uri $detailUrl -Method GET
$strainFiles = Invoke-RestMethod -Uri $detail.strain_files_url -Method GET

$h1Url = Pick-DetectorUrl -StrainFilesJson $strainFiles -Detector "H1" -Duration $Duration -Format $Format
$l1Url = Pick-DetectorUrl -StrainFilesJson $strainFiles -Detector "L1" -Duration $Duration -Format $Format

  if (-not $h1Url -or -not $l1Url) {
    throw "No encontré URLs H1/L1 para $EventId (duration=$Duration format=$Format). detail_url=$detailUrl"
  }

  $h1Path = Download-File -Url $h1Url -OutDir $OutDir
  $l1Path = Download-File -Url $l1Url -OutDir $OutDir

  Write-Host "  [HASH] $EventId"
  Get-FileHash -Algorithm SHA256 $h1Path | Format-Table -AutoSize
  Get-FileHash -Algorithm SHA256 $l1Path | Format-Table -AutoSize
}

Write-Host "[DONE] Descarga finalizada en $OutRoot"
.Exception.Message)"
  }

  # GWOSC puede devolver 'events' (lista) o 'event' (objeto) o estructuras diferentes.
  $detail = $null

  if ($resp -and $resp.PSObject.Properties.Name -contains "events" -and $resp.events -and $resp.events.Count -gt 0) {
    $ev = $resp.events[0]
    if ($ev.versions -and $ev.versions.Count -gt 0) { $detail = $ev.versions[-1].detail_url }
    if (-not $detail -and $ev.detail_url) { $detail = $ev.detail_url }
  } elseif ($resp -and $resp.PSObject.Properties.Name -contains "event" -and $resp.event) {
    $ev = $resp.event
    if ($ev.versions -and $ev.versions.Count -gt 0) { $detail = $ev.versions[-1].detail_url }
    if (-not $detail -and $ev.detail_url) { $detail = $ev.detail_url }
  } elseif ($resp -and $resp.PSObject.Properties.Name -contains "detail_url") {
    $detail = $resp.detail_url
  }

    if (-not $detail -and $resp -and ($resp.PSObject.Properties.Name -contains "versions") -and $resp.versions -and $resp.versions.Count -gt 0) {
    $detail = $resp.versions[-1].detail_url
  }

  if (-not $detail) {
    # Dump mínimo auditable (sin volcar 10k líneas)
    $keys = if ($resp) { ($resp.PSObject.Properties.Name -join ",") } else { "<null>" }
    throw "No detail_url para ${EventId}. API=$api. Top-level keys=[$keys]"
  }
  return $detail
}function Pick-DetectorUrl($DetailJson, [string]$Detector, [int]$Duration, [string]$Format) {
  $strain = $DetailJson.strain | Where-Object { $_.detector -eq $Detector }
  if (-not $strain) { return $null }
  $file = $strain.files | Where-Object { $_.format -eq $Format -and $_.duration -eq $Duration } | Select-Object -First 1
  if (-not $file) { return $null }
  return $file.download_url
}

function Ensure-EventFiles([string]$OutDir) {
  $h1 = Get-ChildItem -Path $OutDir -Filter "*H1*.hdf5" -ErrorAction SilentlyContinue | Select-Object -First 1
  $l1 = Get-ChildItem -Path $OutDir -Filter "*L1*.hdf5" -ErrorAction SilentlyContinue | Select-Object -First 1
  return ($null -ne $h1 -and $null -ne $l1)
}

function Download-File([string]$Url, [string]$OutDir) {
  $fileName = [System.IO.Path]::GetFileName(([uri]$Url).AbsolutePath)
  $outPath = Join-Path $OutDir $fileName

  if (Test-Path $outPath) {
    $len = (Get-Item $outPath).Length
    if ($len -gt 0) {
      Write-Host "  [SKIP] Existe: $outPath ($len bytes)"
      return $outPath
    } else {
      Remove-Item -Force $outPath
    }
  }

  Write-Host "  [GET]  $Url"
  Start-BitsTransfer -Source $Url -Destination $outPath -TransferType Download -ErrorAction Stop
  return $outPath
}

New-Item -ItemType Directory -Force -Path $OutRoot | Out-Null

foreach ($EventId in $Events) {
  $OutDir  = Join-Path $OutRoot $EventId
  New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

  if (Ensure-EventFiles -OutDir $OutDir) {
    Write-Host "[OK]   ${EventId}: ya existe H1+L1 en $OutDir"
    continue
  }

  Write-Host "[MISS] ${EventId}: descargando H1+L1 (duration=$Duration format=$Format) → $OutDir"

  $detailUrl = Get-EventDetailUrl -EventId $EventId
  $detail      = Invoke-RestMethod -Uri $detailUrl -Method GET
$strainFiles = Invoke-RestMethod -Uri $detail.strain_files_url -Method GET

$h1Url = Pick-DetectorUrl -StrainFilesJson $strainFiles -Detector "H1" -Duration $Duration -Format $Format
$l1Url = Pick-DetectorUrl -StrainFilesJson $strainFiles -Detector "L1" -Duration $Duration -Format $Format

  if (-not $h1Url -or -not $l1Url) {
    throw "No encontré URLs H1/L1 para $EventId (duration=$Duration format=$Format). detail_url=$detailUrl"
  }

  $h1Path = Download-File -Url $h1Url -OutDir $OutDir
  $l1Path = Download-File -Url $l1Url -OutDir $OutDir

  Write-Host "  [HASH] $EventId"
  Get-FileHash -Algorithm SHA256 $h1Path | Format-Table -AutoSize
  Get-FileHash -Algorithm SHA256 $l1Path | Format-Table -AutoSize
}

Write-Host "[DONE] Descarga finalizada en $OutRoot"


.format -eq $Format } | Sort-Object duration | Select-Object -First 1
  if ($cand -and $cand.download_url) { return $cand.download_url }

  $cand = $files | Where-Object { $ErrorActionPreference = "Stop"

$RepoRoot = (Get-Location).Path
$OutRoot  = Join-Path $RepoRoot "data\losc"
$Duration = 32
$Format   = "hdf5"

$Events = @(
  "GW150914","GW151226","GW170104","GW170608","GW170729",
  "GW170809","GW170814","GW170818","GW170823","GW190521_030229"
)

function Get-EventDetailUrl([string]$EventId) {
  $api = "https://gwosc.org/api/v2/events/$EventId"
  try {
    $resp = Invoke-RestMethod -Uri $api -Method GET
  } catch {
    throw "GWOSC request failed for ${EventId}. API=$api. $($ErrorActionPreference = "Stop"

$RepoRoot = (Get-Location).Path
$OutRoot  = Join-Path $RepoRoot "data\losc"
$Duration = 32
$Format   = "hdf5"

$Events = @(
  "GW150914","GW151226","GW170104","GW170608","GW170729",
  "GW170809","GW170814","GW170818","GW170823","GW190521_030229"
)

function Get-EventDetailUrl([string]$EventId) {
  $api = "https://gwosc.org/api/v2/events/$EventId"
  $resp = Invoke-RestMethod -Uri $api -Method GET
  $detail = $resp.events[0].versions[-1].detail_url
  if (-not $detail) { throw "No detail_url para $EventId (API=$api)" }
  return $detail
}

function Pick-DetectorUrl($DetailJson, [string]$Detector, [int]$Duration, [string]$Format) {
  $strain = $DetailJson.strain | Where-Object { $_.detector -eq $Detector }
  if (-not $strain) { return $null }
  $file = $strain.files | Where-Object { $_.format -eq $Format -and $_.duration -eq $Duration } | Select-Object -First 1
  if (-not $file) { return $null }
  return $file.download_url
}

function Ensure-EventFiles([string]$OutDir) {
  $h1 = Get-ChildItem -Path $OutDir -Filter "*H1*.hdf5" -ErrorAction SilentlyContinue | Select-Object -First 1
  $l1 = Get-ChildItem -Path $OutDir -Filter "*L1*.hdf5" -ErrorAction SilentlyContinue | Select-Object -First 1
  return ($null -ne $h1 -and $null -ne $l1)
}

function Download-File([string]$Url, [string]$OutDir) {
  $fileName = [System.IO.Path]::GetFileName(([uri]$Url).AbsolutePath)
  $outPath = Join-Path $OutDir $fileName

  if (Test-Path $outPath) {
    $len = (Get-Item $outPath).Length
    if ($len -gt 0) {
      Write-Host "  [SKIP] Existe: $outPath ($len bytes)"
      return $outPath
    } else {
      Remove-Item -Force $outPath
    }
  }

  Write-Host "  [GET]  $Url"
  Start-BitsTransfer -Source $Url -Destination $outPath -TransferType Download -ErrorAction Stop
  return $outPath
}

New-Item -ItemType Directory -Force -Path $OutRoot | Out-Null

foreach ($EventId in $Events) {
  $OutDir  = Join-Path $OutRoot $EventId
  New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

  if (Ensure-EventFiles -OutDir $OutDir) {
    Write-Host "[OK]   ${EventId}: ya existe H1+L1 en $OutDir"
    continue
  }

  Write-Host "[MISS] ${EventId}: descargando H1+L1 (duration=$Duration format=$Format) → $OutDir"

  $detailUrl = Get-EventDetailUrl -EventId $EventId
  $detail      = Invoke-RestMethod -Uri $detailUrl -Method GET
$strainFiles = Invoke-RestMethod -Uri $detail.strain_files_url -Method GET

$h1Url = Pick-DetectorUrl -StrainFilesJson $strainFiles -Detector "H1" -Duration $Duration -Format $Format
$l1Url = Pick-DetectorUrl -StrainFilesJson $strainFiles -Detector "L1" -Duration $Duration -Format $Format

  if (-not $h1Url -or -not $l1Url) {
    throw "No encontré URLs H1/L1 para $EventId (duration=$Duration format=$Format). detail_url=$detailUrl"
  }

  $h1Path = Download-File -Url $h1Url -OutDir $OutDir
  $l1Path = Download-File -Url $l1Url -OutDir $OutDir

  Write-Host "  [HASH] $EventId"
  Get-FileHash -Algorithm SHA256 $h1Path | Format-Table -AutoSize
  Get-FileHash -Algorithm SHA256 $l1Path | Format-Table -AutoSize
}

Write-Host "[DONE] Descarga finalizada en $OutRoot"
.Exception.Message)"
  }

  # GWOSC puede devolver 'events' (lista) o 'event' (objeto) o estructuras diferentes.
  $detail = $null

  if ($resp -and $resp.PSObject.Properties.Name -contains "events" -and $resp.events -and $resp.events.Count -gt 0) {
    $ev = $resp.events[0]
    if ($ev.versions -and $ev.versions.Count -gt 0) { $detail = $ev.versions[-1].detail_url }
    if (-not $detail -and $ev.detail_url) { $detail = $ev.detail_url }
  } elseif ($resp -and $resp.PSObject.Properties.Name -contains "event" -and $resp.event) {
    $ev = $resp.event
    if ($ev.versions -and $ev.versions.Count -gt 0) { $detail = $ev.versions[-1].detail_url }
    if (-not $detail -and $ev.detail_url) { $detail = $ev.detail_url }
  } elseif ($resp -and $resp.PSObject.Properties.Name -contains "detail_url") {
    $detail = $resp.detail_url
  }

    if (-not $detail -and $resp -and ($resp.PSObject.Properties.Name -contains "versions") -and $resp.versions -and $resp.versions.Count -gt 0) {
    $detail = $resp.versions[-1].detail_url
  }

  if (-not $detail) {
    # Dump mínimo auditable (sin volcar 10k líneas)
    $keys = if ($resp) { ($resp.PSObject.Properties.Name -join ",") } else { "<null>" }
    throw "No detail_url para ${EventId}. API=$api. Top-level keys=[$keys]"
  }
  return $detail
}function Pick-DetectorUrl($DetailJson, [string]$Detector, [int]$Duration, [string]$Format) {
  $strain = $DetailJson.strain | Where-Object { $_.detector -eq $Detector }
  if (-not $strain) { return $null }
  $file = $strain.files | Where-Object { $_.format -eq $Format -and $_.duration -eq $Duration } | Select-Object -First 1
  if (-not $file) { return $null }
  return $file.download_url
}

function Ensure-EventFiles([string]$OutDir) {
  $h1 = Get-ChildItem -Path $OutDir -Filter "*H1*.hdf5" -ErrorAction SilentlyContinue | Select-Object -First 1
  $l1 = Get-ChildItem -Path $OutDir -Filter "*L1*.hdf5" -ErrorAction SilentlyContinue | Select-Object -First 1
  return ($null -ne $h1 -and $null -ne $l1)
}

function Download-File([string]$Url, [string]$OutDir) {
  $fileName = [System.IO.Path]::GetFileName(([uri]$Url).AbsolutePath)
  $outPath = Join-Path $OutDir $fileName

  if (Test-Path $outPath) {
    $len = (Get-Item $outPath).Length
    if ($len -gt 0) {
      Write-Host "  [SKIP] Existe: $outPath ($len bytes)"
      return $outPath
    } else {
      Remove-Item -Force $outPath
    }
  }

  Write-Host "  [GET]  $Url"
  Start-BitsTransfer -Source $Url -Destination $outPath -TransferType Download -ErrorAction Stop
  return $outPath
}

New-Item -ItemType Directory -Force -Path $OutRoot | Out-Null

foreach ($EventId in $Events) {
  $OutDir  = Join-Path $OutRoot $EventId
  New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

  if (Ensure-EventFiles -OutDir $OutDir) {
    Write-Host "[OK]   ${EventId}: ya existe H1+L1 en $OutDir"
    continue
  }

  Write-Host "[MISS] ${EventId}: descargando H1+L1 (duration=$Duration format=$Format) → $OutDir"

  $detailUrl = Get-EventDetailUrl -EventId $EventId
  $detail      = Invoke-RestMethod -Uri $detailUrl -Method GET
$strainFiles = Invoke-RestMethod -Uri $detail.strain_files_url -Method GET

$h1Url = Pick-DetectorUrl -StrainFilesJson $strainFiles -Detector "H1" -Duration $Duration -Format $Format
$l1Url = Pick-DetectorUrl -StrainFilesJson $strainFiles -Detector "L1" -Duration $Duration -Format $Format

  if (-not $h1Url -or -not $l1Url) {
    throw "No encontré URLs H1/L1 para $EventId (duration=$Duration format=$Format). detail_url=$detailUrl"
  }

  $h1Path = Download-File -Url $h1Url -OutDir $OutDir
  $l1Path = Download-File -Url $l1Url -OutDir $OutDir

  Write-Host "  [HASH] $EventId"
  Get-FileHash -Algorithm SHA256 $h1Path | Format-Table -AutoSize
  Get-FileHash -Algorithm SHA256 $l1Path | Format-Table -AutoSize
}

Write-Host "[DONE] Descarga finalizada en $OutRoot"


.detector -eq $Detector -and $ErrorActionPreference = "Stop"

$RepoRoot = (Get-Location).Path
$OutRoot  = Join-Path $RepoRoot "data\losc"
$Duration = 32
$Format   = "hdf5"

$Events = @(
  "GW150914","GW151226","GW170104","GW170608","GW170729",
  "GW170809","GW170814","GW170818","GW170823","GW190521_030229"
)

function Get-EventDetailUrl([string]$EventId) {
  $api = "https://gwosc.org/api/v2/events/$EventId"
  try {
    $resp = Invoke-RestMethod -Uri $api -Method GET
  } catch {
    throw "GWOSC request failed for ${EventId}. API=$api. $($ErrorActionPreference = "Stop"

$RepoRoot = (Get-Location).Path
$OutRoot  = Join-Path $RepoRoot "data\losc"
$Duration = 32
$Format   = "hdf5"

$Events = @(
  "GW150914","GW151226","GW170104","GW170608","GW170729",
  "GW170809","GW170814","GW170818","GW170823","GW190521_030229"
)

function Get-EventDetailUrl([string]$EventId) {
  $api = "https://gwosc.org/api/v2/events/$EventId"
  $resp = Invoke-RestMethod -Uri $api -Method GET
  $detail = $resp.events[0].versions[-1].detail_url
  if (-not $detail) { throw "No detail_url para $EventId (API=$api)" }
  return $detail
}

function Pick-DetectorUrl($DetailJson, [string]$Detector, [int]$Duration, [string]$Format) {
  $strain = $DetailJson.strain | Where-Object { $_.detector -eq $Detector }
  if (-not $strain) { return $null }
  $file = $strain.files | Where-Object { $_.format -eq $Format -and $_.duration -eq $Duration } | Select-Object -First 1
  if (-not $file) { return $null }
  return $file.download_url
}

function Ensure-EventFiles([string]$OutDir) {
  $h1 = Get-ChildItem -Path $OutDir -Filter "*H1*.hdf5" -ErrorAction SilentlyContinue | Select-Object -First 1
  $l1 = Get-ChildItem -Path $OutDir -Filter "*L1*.hdf5" -ErrorAction SilentlyContinue | Select-Object -First 1
  return ($null -ne $h1 -and $null -ne $l1)
}

function Download-File([string]$Url, [string]$OutDir) {
  $fileName = [System.IO.Path]::GetFileName(([uri]$Url).AbsolutePath)
  $outPath = Join-Path $OutDir $fileName

  if (Test-Path $outPath) {
    $len = (Get-Item $outPath).Length
    if ($len -gt 0) {
      Write-Host "  [SKIP] Existe: $outPath ($len bytes)"
      return $outPath
    } else {
      Remove-Item -Force $outPath
    }
  }

  Write-Host "  [GET]  $Url"
  Start-BitsTransfer -Source $Url -Destination $outPath -TransferType Download -ErrorAction Stop
  return $outPath
}

New-Item -ItemType Directory -Force -Path $OutRoot | Out-Null

foreach ($EventId in $Events) {
  $OutDir  = Join-Path $OutRoot $EventId
  New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

  if (Ensure-EventFiles -OutDir $OutDir) {
    Write-Host "[OK]   ${EventId}: ya existe H1+L1 en $OutDir"
    continue
  }

  Write-Host "[MISS] ${EventId}: descargando H1+L1 (duration=$Duration format=$Format) → $OutDir"

  $detailUrl = Get-EventDetailUrl -EventId $EventId
  $detail      = Invoke-RestMethod -Uri $detailUrl -Method GET
$strainFiles = Invoke-RestMethod -Uri $detail.strain_files_url -Method GET

$h1Url = Pick-DetectorUrl -StrainFilesJson $strainFiles -Detector "H1" -Duration $Duration -Format $Format
$l1Url = Pick-DetectorUrl -StrainFilesJson $strainFiles -Detector "L1" -Duration $Duration -Format $Format

  if (-not $h1Url -or -not $l1Url) {
    throw "No encontré URLs H1/L1 para $EventId (duration=$Duration format=$Format). detail_url=$detailUrl"
  }

  $h1Path = Download-File -Url $h1Url -OutDir $OutDir
  $l1Path = Download-File -Url $l1Url -OutDir $OutDir

  Write-Host "  [HASH] $EventId"
  Get-FileHash -Algorithm SHA256 $h1Path | Format-Table -AutoSize
  Get-FileHash -Algorithm SHA256 $l1Path | Format-Table -AutoSize
}

Write-Host "[DONE] Descarga finalizada en $OutRoot"
.Exception.Message)"
  }

  # GWOSC puede devolver 'events' (lista) o 'event' (objeto) o estructuras diferentes.
  $detail = $null

  if ($resp -and $resp.PSObject.Properties.Name -contains "events" -and $resp.events -and $resp.events.Count -gt 0) {
    $ev = $resp.events[0]
    if ($ev.versions -and $ev.versions.Count -gt 0) { $detail = $ev.versions[-1].detail_url }
    if (-not $detail -and $ev.detail_url) { $detail = $ev.detail_url }
  } elseif ($resp -and $resp.PSObject.Properties.Name -contains "event" -and $resp.event) {
    $ev = $resp.event
    if ($ev.versions -and $ev.versions.Count -gt 0) { $detail = $ev.versions[-1].detail_url }
    if (-not $detail -and $ev.detail_url) { $detail = $ev.detail_url }
  } elseif ($resp -and $resp.PSObject.Properties.Name -contains "detail_url") {
    $detail = $resp.detail_url
  }

    if (-not $detail -and $resp -and ($resp.PSObject.Properties.Name -contains "versions") -and $resp.versions -and $resp.versions.Count -gt 0) {
    $detail = $resp.versions[-1].detail_url
  }

  if (-not $detail) {
    # Dump mínimo auditable (sin volcar 10k líneas)
    $keys = if ($resp) { ($resp.PSObject.Properties.Name -join ",") } else { "<null>" }
    throw "No detail_url para ${EventId}. API=$api. Top-level keys=[$keys]"
  }
  return $detail
}function Pick-DetectorUrl($DetailJson, [string]$Detector, [int]$Duration, [string]$Format) {
  $strain = $DetailJson.strain | Where-Object { $_.detector -eq $Detector }
  if (-not $strain) { return $null }
  $file = $strain.files | Where-Object { $_.format -eq $Format -and $_.duration -eq $Duration } | Select-Object -First 1
  if (-not $file) { return $null }
  return $file.download_url
}

function Ensure-EventFiles([string]$OutDir) {
  $h1 = Get-ChildItem -Path $OutDir -Filter "*H1*.hdf5" -ErrorAction SilentlyContinue | Select-Object -First 1
  $l1 = Get-ChildItem -Path $OutDir -Filter "*L1*.hdf5" -ErrorAction SilentlyContinue | Select-Object -First 1
  return ($null -ne $h1 -and $null -ne $l1)
}

function Download-File([string]$Url, [string]$OutDir) {
  $fileName = [System.IO.Path]::GetFileName(([uri]$Url).AbsolutePath)
  $outPath = Join-Path $OutDir $fileName

  if (Test-Path $outPath) {
    $len = (Get-Item $outPath).Length
    if ($len -gt 0) {
      Write-Host "  [SKIP] Existe: $outPath ($len bytes)"
      return $outPath
    } else {
      Remove-Item -Force $outPath
    }
  }

  Write-Host "  [GET]  $Url"
  Start-BitsTransfer -Source $Url -Destination $outPath -TransferType Download -ErrorAction Stop
  return $outPath
}

New-Item -ItemType Directory -Force -Path $OutRoot | Out-Null

foreach ($EventId in $Events) {
  $OutDir  = Join-Path $OutRoot $EventId
  New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

  if (Ensure-EventFiles -OutDir $OutDir) {
    Write-Host "[OK]   ${EventId}: ya existe H1+L1 en $OutDir"
    continue
  }

  Write-Host "[MISS] ${EventId}: descargando H1+L1 (duration=$Duration format=$Format) → $OutDir"

  $detailUrl = Get-EventDetailUrl -EventId $EventId
  $detail      = Invoke-RestMethod -Uri $detailUrl -Method GET
$strainFiles = Invoke-RestMethod -Uri $detail.strain_files_url -Method GET

$h1Url = Pick-DetectorUrl -StrainFilesJson $strainFiles -Detector "H1" -Duration $Duration -Format $Format
$l1Url = Pick-DetectorUrl -StrainFilesJson $strainFiles -Detector "L1" -Duration $Duration -Format $Format

  if (-not $h1Url -or -not $l1Url) {
    throw "No encontré URLs H1/L1 para $EventId (duration=$Duration format=$Format). detail_url=$detailUrl"
  }

  $h1Path = Download-File -Url $h1Url -OutDir $OutDir
  $l1Path = Download-File -Url $l1Url -OutDir $OutDir

  Write-Host "  [HASH] $EventId"
  Get-FileHash -Algorithm SHA256 $h1Path | Format-Table -AutoSize
  Get-FileHash -Algorithm SHA256 $l1Path | Format-Table -AutoSize
}

Write-Host "[DONE] Descarga finalizada en $OutRoot"


.download_url } | Sort-Object format,duration | Select-Object -First 1
  if ($cand -and $cand.download_url) { return $cand.download_url }

  return $null
}function Ensure-EventFiles([string]$OutDir) {
  $h1 = Get-ChildItem -Path $OutDir -Filter "*H1*.hdf5" -ErrorAction SilentlyContinue | Select-Object -First 1
  $l1 = Get-ChildItem -Path $OutDir -Filter "*L1*.hdf5" -ErrorAction SilentlyContinue | Select-Object -First 1
  return ($null -ne $h1 -and $null -ne $l1)
}

function Download-File([string]$Url, [string]$OutDir) {
  $fileName = [System.IO.Path]::GetFileName(([uri]$Url).AbsolutePath)
  $outPath = Join-Path $OutDir $fileName

  if (Test-Path $outPath) {
    $len = (Get-Item $outPath).Length
    if ($len -gt 0) {
      Write-Host "  [SKIP] Existe: $outPath ($len bytes)"
      return $outPath
    } else {
      Remove-Item -Force $outPath
    }
  }

  Write-Host "  [GET]  $Url"
  Start-BitsTransfer -Source $Url -Destination $outPath -TransferType Download -ErrorAction Stop
  return $outPath
}

New-Item -ItemType Directory -Force -Path $OutRoot | Out-Null

foreach ($EventId in $Events) {
  $OutDir  = Join-Path $OutRoot $EventId
  New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

  if (Ensure-EventFiles -OutDir $OutDir) {
    Write-Host "[OK]   ${EventId}: ya existe H1+L1 en $OutDir"
    continue
  }

  Write-Host "[MISS] ${EventId}: descargando H1+L1 (duration=$Duration format=$Format) → $OutDir"

  $detailUrl = Get-EventDetailUrl -EventId $EventId
  $detail      = Invoke-RestMethod -Uri $detailUrl -Method GET
$strainFiles = Invoke-RestMethod -Uri $detail.strain_files_url -Method GET

$h1Url = Pick-DetectorUrl -StrainFilesJson $strainFiles -Detector "H1" -Duration $Duration -Format $Format
$l1Url = Pick-DetectorUrl -StrainFilesJson $strainFiles -Detector "L1" -Duration $Duration -Format $Format

  if (-not $h1Url -or -not $l1Url) {
    throw "No encontré URLs H1/L1 para $EventId (duration=$Duration format=$Format). detail_url=$detailUrl"
  }

  $h1Path = Download-File -Url $h1Url -OutDir $OutDir
  $l1Path = Download-File -Url $l1Url -OutDir $OutDir

  Write-Host "  [HASH] $EventId"
  Get-FileHash -Algorithm SHA256 $h1Path | Format-Table -AutoSize
  Get-FileHash -Algorithm SHA256 $l1Path | Format-Table -AutoSize
}

Write-Host "[DONE] Descarga finalizada en $OutRoot"




