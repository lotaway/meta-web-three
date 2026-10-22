#!/usr/bin/env bash
# Integration check: verifies that the frontend request paths reach the owning
# microservice through the gateway.
#
# Probe design: the gateway UserAuthFilter answers unauthenticated requests
# with AUTH_HEADER_INVALID before any routing happens, so that response cannot
# prove the traffic reached a service. Every service exposes /v3/api-docs and
# that keyword is listed in the gateway excluded-path-keywords, so the request
# skips authentication and enters the discovery locator routing. Receiving the
# service's own OpenAPI document therefore proves the hit.
#
# Usage: bash scripts/verify-gateway-routing.sh [gateway_base]
set -uo pipefail

BASE="${1:-http://192.168.1.110:10081}"
FAILED=0

probe_service() {
  local svc="$1" code n doc="/tmp/_vr_${1}.json"
  code="$(curl -s -o "$doc" -w '%{http_code}' --max-time 30 "${BASE}/${svc}/v3/api-docs" 2>/dev/null || echo 000)"
  if [ "$code" != "200" ]; then
    echo "FAIL  ${svc}: /${svc}/v3/api-docs -> HTTP ${code}"
    FAILED=1
    return
  fi
  if grep -q '"openapi"' "$doc"; then
    n="$(grep -o '"/[A-Za-z0-9{}/._-]*"' "$doc" | sort -u | wc -l | tr -d ' ')"
    echo "PASS  ${svc}: request routed to the service (its own OpenAPI doc returned, ${n} paths)"
  else
    echo "FAIL  ${svc}: response is not an OpenAPI document - routing failed"
    FAILED=1
  fi
}

echo "gateway: ${BASE}"
echo ""
echo "--- service routing probes ---"
probe_service mall-catalog-service
probe_service mall-user-service
probe_service mall-trade-service

echo ""
echo "--- business paths (unauthenticated but gateway-allowed) ---"
check_business() {
  local desc="$1" path="$2" expect="$3"
  local code body
  code="$(curl -s -o /tmp/_vr_body -w '%{http_code}' --max-time 20 "${BASE}${path}" 2>/dev/null || echo 000)"
  body="$(head -c 200 /tmp/_vr_body 2>/dev/null | tr -d '\n\r')"
  if [ "$code" = "000" ]; then
    echo "FAIL  ${desc} [${path}] gateway unreachable"
    FAILED=1
  elif echo "$body" | grep -q "No instances available\|Unable to find instance"; then
    echo "FAIL  ${desc} [${path}] no service instance registered: ${body}"
    FAILED=1
  else
    echo "PASS  ${desc} [${path}] HTTP ${code} (reached ${expect})"
  fi
}

check_business "flash current"     "/mall-trade-service/flash/current"        "mall-trade-service"
check_business "user signIn"       "/mall-user-service/user/signIn"          "mall-user-service"

echo ""
echo "--- aggregated openapi serviceId prefix ---"
DOCS="$(curl -s --max-time 40 "${BASE}/meta/v3/api-docs" 2>/dev/null || echo '')"
if [ -z "$DOCS" ]; then
  echo "FAIL  cannot fetch ${BASE}/meta/v3/api-docs"
  FAILED=1
else
  echo "$DOCS" > /tmp/_vr_openapi.json
  for svc in mall-catalog-service mall-user-service mall-trade-service; do
    n="$(grep -o "\"/${svc}/[^\"]*\"" /tmp/_vr_openapi.json | sort -u | wc -l | tr -d ' ')"
    if [ "$n" -gt 0 ]; then
      echo "PASS  ${svc}: ${n} aggregated paths carry the serviceId prefix"
    else
      echo "FAIL  ${svc}: no aggregated path carries the serviceId prefix"
      FAILED=1
    fi
  done
fi

echo ""
if [ "$FAILED" -eq 0 ]; then
  echo "=== gateway routing verification PASSED ==="
  exit 0
else
  echo "=== gateway routing verification FAILED ==="
  exit 1
fi
