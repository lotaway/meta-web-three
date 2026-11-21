# Sets the GitHub Actions repository variable used by the performance pipeline.
# Usage:
#   .\scripts\set-perf-base-url.ps1 -Url "http://<staging-gateway-host>:10081"
#
# Reads the GitHub credential from the local git credential manager; never
# prints the token. Requires admin/maintain permission on the repository.
param(
    [Parameter(Mandatory = $true)]
    [string] $Url,
    [string] $Repo = 'lotaway/meta-web-three'
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
Write-Output "Verified: PERF_BASE_URL=$($check.value)"
