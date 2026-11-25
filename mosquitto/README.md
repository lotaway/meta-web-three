# Local MQTT broker (dev/test only)

Started by `docker compose -f docker-compose.env.yml up -d mqtt`. It publishes
no data by itself — telemetry only flows when a service with `MQTT_ENABLED=true`
subscribes and a producer publishes.

## Config

- `mosquitto.conf`: anonymous listener on 1883 (dev/test only; production must
  mount an authenticated config and point `MQTT_BROKER_URL` at the real broker).
- All addresses/ports are env-injected in `docker-compose.env.yml` / `.env.example`:
  `MQTT_IMAGE`, `MQTT_PORT`, `MQTT_BROKER_URL`, `MQTT_TOPIC_PREFIX`. Nothing is
  hard-coded in Java.

## Publishing test data (explicit opt-in)

```bash
.\scripts\test-mqtt-telemetry.ps1 -BrokerContainer default-mqtt -Device AGV-001 -Count 5
```

This script is never called automatically. It publishes to the normal topic
prefix (`mes/equipment/<device>/telemetry`) so a service with MQTT enabled can
ingest it. Run it only in dev/test.

## Java integration test (opt-in)

`server/factory-domain/factory-service/.../MqttTelemetryIntegrationTest.java`
uses an **isolated topic prefix** (`itest/`, via `MQTT_IT_PREFIX`) so its data can
never enter a production topic. Run only where the broker is reachable from the
JVM (e.g. host port mapping present):

```bash
MQTT_IT=1 MQTT_IT_BROKER=tcp://localhost:1883 mvn -pl factory-domain/factory-service -am test -Dtest=MqttTelemetryIntegrationTest
```

It is skipped by default (no `MQTT_IT`), so the normal unit-test suite is unaffected.
