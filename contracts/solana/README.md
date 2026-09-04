# Solana Contract — Local Development & Testing

Local testing for the `solana-contract` Anchor program uses **surfpool** as the local
Solana network. The workflow has only three steps: start the local service, run the
TS tests, then stop the local service.

## Prerequisites

- RPC port is **18999** (WS 19000). This machine's WSL2 mirrored mode reserves the host
  TCP range 8000–15000, so the default 8899 cannot be bound.
- Surfpool runs online against a remote datasource to lazy-load the `mpl-token-metadata`
  program. It inherits the shell's global proxy; no proxy config is needed.
- `NO_DNA=1` is required, otherwise surfpool enters interactive/TUI mode and fails to
  reach the datasource.

## 1. Start the local service

Run in a dedicated terminal (foreground; `--watch` auto re-deploys when `.so` files change):

```bash
NO_DNA=1 surfpool start --network devnet \
  --port 18999 --ws-port 19000 \
  --no-tui --no-studio --yes --watch
```

The program is deployed automatically on startup. Leave this terminal running while
developing.

## 2. Run the tests

Run the TS test script directly against the running surfpool (tests use `describe`/`it`,
so they run under `ts-mocha`, matching the `Anchor.toml` test script):

```bash
ANCHOR_PROVIDER_URL=http://127.0.0.1:18999 \
  npx ts-mocha -p ./tsconfig.json -t 1000000 tests/solana-contract.ts
```

For custom TS scripts, point them at the local ledger the same way and run with
`npx ts-node`:

```bash
ANCHOR_PROVIDER_URL=http://127.0.0.1:18999 ANCHOR_WALLET=~/.config/solana/id.json \
  npx ts-node tests/your-script.ts
```

## 3. Stop the local service

Return to the surfpool terminal and press `Ctrl+C`.

Surfpool also responds to `SIGTERM` for a graceful shutdown:

```bash
pkill -TERM -x surfpool
```

Confirm it stopped:

```bash
pgrep -x surfpool   # no output means it stopped
```

## Notes

- `--network` only selects the remote datasource for lazy-loading programs. Tests
  always run on the local ledger at `127.0.0.1:18999`.
- Intermittent `Internal error` / `Failed to fetch accounts from remote` test failures
  are usually transient proxy issues with the remote datasource — re-run to confirm.
