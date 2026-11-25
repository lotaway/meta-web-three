# Publishes sample telemetry to an MQTT broker for LOCAL/DEV testing only.
# Nothing calls this automatically - run it explicitly when you want to verify
# the digital-twin telemetry path. In production, real devices publish instead.
#
# Usage:
#   .\scripts\test-mqtt-telemetry.ps1
#   .\scripts\test-mqtt-telemetry.ps1 -BrokerContainer default-mqtt -Device AGV-001 -Count 10
param(
    [string] $BrokerContainer = 'default-mqtt',
    [string] $TopicPrefix = 'mes/equipment/',
    [string] $Device = 'AGV-001',
    [int] $Count = 5,
    [int] $IntervalMs = 2000
)

$ErrorActionPreference = 'Stop'

if (-not (docker ps --format '{{.Names}}' | Select-String -Pattern "^$BrokerContainer$")) {
    throw "Broker container '$BrokerContainer' is not running. Start the compose stack first."
}

$topic = "${TopicPrefix}${Device}/telemetry"
Write-Output "Publishing $Count samples to $topic"

for ($i = 1; $i -le $Count; $i++) {
    $timestamp = [DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds()
    $temperature = [math]::Round(20 + (Get-Random -Minimum 0 -Maximum 100) / 10, 1)
    $vibration = [math]::Round((Get-Random -Minimum 0 -Maximum 100) / 100, 2)
    # Payload shape matches mes.telemetry mapping: timestamp/quality fields + metric fields.
    $payload = @{
        timestamp = $timestamp
        quality = 'GOOD'
        temperature = $temperature
        vibration = $vibration
        unit = @{ temperature = 'C'; vibration = 'mm/s' }
    } | ConvertTo-Json -Compress

    docker exec $BrokerContainer mosquitto_pub -h localhost -t $topic -m $payload
    Write-Output "  [$i/$Count] $payload"
    if ($i -lt $Count) { Start-Sleep -Milliseconds $IntervalMs }
}

Write-Output "Done. Telemetry is now in the broker; enable MQTT_ENABLED on factory-service to ingest it."
