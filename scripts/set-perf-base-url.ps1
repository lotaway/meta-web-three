# Configure and verify the performance pipeline's target.
#
# Usage:
#   .\scripts\set-perf-base-url.ps1 -Url "http://<staging-gateway-host>:10081"
#
# The URL must be reachable from GitHub's ubuntu-24.04 runner (i.e. a real
# staging host, not a private/LAN address). Reads the GitHub credential from
# the local git credential manager; never prints the token. Requires admin or
# maintain permission on the repository.
param(
    [Parameter(Mandatory = $true)]
    [string] $Url,
    [string] $Repo = 'lotaway/meta-web-three',
    [switch] $Dispatch,
    [int] $WaitSeconds = 180
)

$ErrorActionPreference = 'Stop'

$credentialInput = "protocol=https`nhost=github.com`n`n"
$credentialOutput = $credentialInput | git credential fill 2>$null
$token = ''
foreach ($line in $credentialOutput) {
    if ($line.StartsWith('password=')) { $token = $line.Substring(9) }
}
if (-not $token) { throw 'No GitHub credential found in git credential manager.' }

$headers = @{
    Authorization = "Bearer $token"
    'User-Agent' = 'opencode-agent'
    Accept = 'application/vnd.github+json'
}

# Upsert: try PATCH first, fall back to POST when the variable does not exist yet.
$body = @{ name = 'PERF_BASE_URL'; value = $Url } | ConvertTo-Json -Compress
try {
    Invoke-RestMethod -Uri "https://api.github.com/repos/$Repo/actions/variables/PERF_BASE_URL" `
        -Method Patch -Headers $headers -Body $body -TimeoutSec 20 | Out-Null
    Write-Output "Updated PERF_BASE_URL for $Repo"
} catch {
    Invoke-RestMethod -Uri "https://api.github.com/repos/$Repo/actions/variables" `
        -Method Post -Headers $headers -Body $body -TimeoutSec 20 | Out-Null
    Write-Output "Created PERF_BASE_URL for $Repo"
}

$check = Invoke-RestMethod -Uri "https://api.github.com/repos/$Repo/actions/variables/PERF_BASE_URL" `
    -Headers $headers -TimeoutSec 20
Write-Output "Verified variable: PERF_BASE_URL=$($check.value)"

if (-not $Dispatch) {
    Write-Output 'Done. Re-run with -Dispatch to trigger and verify a live run.'
    return
}

# Trigger a manual workflow_dispatch run, then wait for it to finish.
$dispatchBody = @{ ref = 'master'; inputs = @{ test_suite = 'quick'; environment = 'staging'; threads = '5'; duration = '60' } } | ConvertTo-Json -Compress
Invoke-RestMethod -Uri "https://api.github.com/repos/$Repo/actions/workflows/performance-test.yml/dispatches" `
    -Method Post -Headers $headers -Body $dispatchBody -TimeoutSec 20 | Out-Null
Write-Output 'Dispatched performance-test workflow.'

$deadline = (Get-Date).AddSeconds($WaitSeconds)
$run = $null
while ((Get-Date) -lt $deadline) {
    Start-Sleep -Seconds 10
    $runs = Invoke-RestMethod -Uri "https://api.github.com/repos/$Repo/actions/workflows/performance-test.yml/runs?per_page=1&event=workflow_dispatch" `
        -Headers $headers -TimeoutSec 20
    $run = $runs.workflow_runs[0]
    Write-Output "run $($run.id): status=$($run.status) conclusion=$($run.conclusion)"
    if ($run.status -eq 'completed') { break }
}

if (-not $run -or $run.status -ne 'completed') {
    Write-Warning "Run did not complete within ${WaitSeconds}s: $($run.html_url)"
    return
}
Write-Output "Run finished: $($run.html_url)"
Write-Output 'Check the run log for: "JMeter target: <your URL>" and absence of MOCK DATA.'
