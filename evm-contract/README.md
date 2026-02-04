# EVM Contracts

This package uses two test stacks:

- Hardhat (JavaScript/TypeScript) for integration and end-to-end flows.
- Foundry (Solidity) for unit tests, fuzz tests, and invariants.

## Install

```bash
yarn install
```

## Testing

Run everything:

```bash
yarn test
```

Run Hardhat tests only:

```bash
yarn test:hardhat
```

Run Foundry tests only:

```bash
yarn test:forge
```

## Foundry Setup

Foundry is not an npm dependency. Install it once on your machine:

```bash
curl -L https://foundry.paradigm.xyz | bash
foundryup
```

Verify it is installed:

```bash
forge --version
```

### CI notes

If CI runs `yarn test`, ensure Foundry is installed in CI before running tests.
