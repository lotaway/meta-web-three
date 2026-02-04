# Commission Service

## Endpoints

- `POST /v1/commission/relations/bind`
- `POST /v1/commission/calc`
- `POST /v1/commission/settle`
- `POST /v1/commission/cancel`
- `GET /v1/commission/balance`
- `GET /v1/commission/ledger`

## Config

```yaml
commission:
  buy-rate: "0.10"
  level-rates: "0.4,0.2,0.1"
  max-levels: 3
  return-window-days: 7
```

## Schema

See `src/main/resources/db/init.sql`.
