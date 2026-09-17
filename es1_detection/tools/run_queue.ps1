# Esegue in sequenza una coda di esperimenti (uno alla volta sulla GPU).
# Il codice di src/ viene copiato per OGNI esperimento al momento della creazione
# della coda, quindi modifiche successive a src/ non toccano la coda.
# Uso: run_queue.ps1 -QueueFile coda.json [-WaitPid 1234]
#   coda.json = [{"name": "e1_anchor8", "overrides": {...}}, ...]
param(
    [Parameter(Mandatory = $true)][string]$QueueFile,
    [int]$WaitPid = 0,
    [switch]$Launch   # uso interno: esecuzione vera, in un processo separato
)
$root = "C:\Users\matte\IdeaProjects\Deep_Learning\es1_detection"
$python = Join-Path $root "venv\Scripts\python.exe"
$queue = Get-Content $QueueFile -Raw | ConvertFrom-Json

if (-not $Launch) {
    # fase 1: snapshot del codice per ogni esperimento, poi avvio in background
    foreach ($exp in $queue) {
        $out = Join-Path $root "outputs\$($exp.name)"
        if (Test-Path (Join-Path $out "history.json")) { throw "$($exp.name) esiste gia': scegli un altro nome" }
        New-Item -ItemType Directory -Force $out | Out-Null
        $snap = Join-Path $out "src"
        if (Test-Path $snap) { Remove-Item -Recurse -Force $snap -Confirm:$false }
        Copy-Item -Recurse (Join-Path $root "src") $snap
        Get-ChildItem $snap -Recurse -Directory -Filter __pycache__ | Remove-Item -Recurse -Force -Confirm:$false
        Write-Output "snapshot: $($exp.name)"
    }
    $self = $MyInvocation.MyCommand.Path
    $p = Start-Process powershell -WindowStyle Hidden -PassThru -ArgumentList @(
        "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", "`"$self`"",
        "-QueueFile", "`"$QueueFile`"", "-WaitPid", $WaitPid, "-Launch")
    Write-Output "coda avviata (PID $($p.Id)), $($queue.Count) esperimenti"
    exit 0
}

# fase 2: esecuzione sequenziale
$qlog = Join-Path $root "outputs\queue.log"
if ($WaitPid -gt 0) {
    Add-Content $qlog "$(Get-Date -Format s) attendo la fine del PID $WaitPid"
    while (Get-Process -Id $WaitPid -ErrorAction SilentlyContinue) { Start-Sleep -Seconds 20 }
}
foreach ($exp in $queue) {
    $out = Join-Path $root "outputs\$($exp.name)"
    $cfg = if ($exp.overrides) { $exp.overrides } else { [pscustomobject]@{} }
    $cfg | Add-Member -NotePropertyName EXPERIMENT -NotePropertyValue $exp.name -Force
    $env:ES1_CONFIG = ($cfg | ConvertTo-Json -Compress -Depth 5)
    $env:ES1_ROOT = $root
    $env:PYTHONUNBUFFERED = "1"
    $env:PYTHONWARNINGS = "ignore"
    Add-Content $qlog "$(Get-Date -Format s) START $($exp.name) $env:ES1_CONFIG"
    $p = Start-Process -FilePath $python -ArgumentList @((Join-Path $out "src\train.py")) `
        -WorkingDirectory $root -WindowStyle Hidden -PassThru -Wait `
        -RedirectStandardOutput (Join-Path $out "train.log") -RedirectStandardError (Join-Path $out "train.err")
    Add-Content $qlog "$(Get-Date -Format s) END   $($exp.name) exit=$($p.ExitCode)"
}
Add-Content $qlog "$(Get-Date -Format s) CODA COMPLETATA"
