# Development Guide

## AI Agent Workflow

1. **Requirement → Task Breakdown**: Decompose requirements into actionable tasks in [TODO.md](./TODO.md)
2. **Task Format**: `[ ]` for incomplete, `[x]` for completed
3. **Development**: Implement tasks sequentially following the specifications below
4. **Quality Check**: After completion, run [CHECK_LIST.md](./CHECK_LIST.md) to verify each task:
   - If passed → remove the task item from TODO.md
   - If failed → mark as `[ ]` and append issue description with fix suggestions
5. **Repeat**: Continue with next task until all are done

## End-to-End Development Flow

```
Requirement → TODO.md breakdown
  → Data structure (DB schema / proto)
  → Backend service + controller
  → Generate API client (OpenAPI → TypeScript)
  → Frontend pages (admin / client / digital-twin)
  → Test
  → CHECK_LIST.md verification
  → Update TODO.md
```

### 1. Data Structure Design
- Define DB schemas in `src/main/resources/db/schema.sql`
- Add migration scripts in `src/main/resources/db/migration/`
- For cross-service interfaces, define `.proto` in `protos/` and run `make gen`
- See [server/README.md](./server/README.md#id-生成策略) for ID strategy and conventions

### 2. Backend Implementation
- Implement service + controller in the appropriate domain module
- Follow [Backend Code Principles](./CODE_PINCEPLES/CODE_PRICEPLES)
- For new services, follow the [new service checklist](./server/README.md#添加新服务清单)
- Ensure the controller is annotated with SpringDoc/Springfox for OpenAPI spec generation

### 3. Frontend API Client

Both apps call backend APIs through the gateway on port `10081` (see
`scripts/server-services-registry.sh`), using a `/{serviceId}` prefix that the
gateway's discovery locator routes to the target service. The two apps use
different env prefixes and different API-client strategies; these are
intentional and documented below, not inconsistencies to unify.

**[backstage-admin](apps/backstage-admin/) — hand-written API modules**

- Env prefix is `VITE_*` (Vite only inlines `VITE_*`). Configure `.env` by
  copying `.env.example`; `src/utils/env.ts` supplies defaults so every
  variable is optional. The generator reads `VITE_BASE_SERVER_URL` and
  `VITE_API_DOC_HOST` from `.env` / `.env.local`.
- The app imports 93 hand-written modules from `src/apis/`, not a generated
  client. `yarn generate:api` (shared `apps/tools/OpenApiToTS.js`) can emit
  `src/generated/api/`, but that output is **deliberately not consumed** — the
  hand-written modules are the stable public interface for all views.
- Sync rule: generated output is a reference for controller signatures only.
  When a backend endpoint changes, update the matching `src/apis/*.ts` module
  and keep `SERVICE_PREFIX_MAP` in `src/utils/http.ts` in sync; generated files
  are never imported at runtime. See
  [apps/backstage-admin/README.md](./apps/backstage-admin/README.md).

**[customer client](apps/client/) — generated API client**

- The generator reads `NEXT_PUBLIC_BACK_API_HOST` / `NEXT_PUBLIC_BACK_API_DOC_HOST`
  from `.env` (then `.env.development` / `.env.local`) and emits both REST and
  GraphQL clients to `src/generated/api/` (and `src/generated/graphql/`).
  `NEXT_PUBLIC_*` here is a code-generation / web-build variable only.
- `src/generated/` is gitignored (React Native cannot commit the generated
  output here). A fresh checkout must run `yarn generate:api` with a reachable
  gateway before `tsc` / tests pass — `api/generated.ts` imports from
  `@/src/generated/api`.
- At runtime the app reads `EXPO_PUBLIC_BACK_API_HOST` (Metro only inlines
  `EXPO_PUBLIC_*`); when unset it is derived from the dev-server `hostUri`
  (Android emulator `10.0.2.2`, iOS simulator `localhost`).

**Prerequisites (generator route):** a running gateway exposing the aggregated
OpenAPI doc at `{host}/meta/v3/api-docs`, and
`npm install -g @openapitools/openapi-generator-cli`.
See [apps/client/README.md](./apps/client/README.md#generate-api-interfaces)
and [apps/backstage-admin/README.md](./apps/backstage-admin/README.md#api-client) for full details.

> The digital-twin desktop app now lives in its own repository (`lotaway/meta-note`) and is no longer part of this monorepo. Its backend (`server/gateway`, `server/factory-domain/factory-service`) still lives here.

### 4. Frontend Pages
- [Backstage Admin](apps/backstage-admin/) — enterprise admin panel (Vue 3 + Element Plus)
- [Customer Client App](apps/client/) — customer mobile app (React Native / Expo)
- Follow [Frontend Code Principles](./CODE_PINCEPLES/FRONTEND_PRICEPLES)

### 5. Testing
- Backend: unit/integration tests in `src/test/` following existing patterns
- Frontend: add tests matching the project's test framework

## Code Principles

All code must follow the principle files in [CODE_PINCEPLES/](./CODE_PINCEPLES/):
- [Backend Code Principles](./CODE_PINCEPLES/CODE_PRICEPLES)
- [Frontend Code Principles](./CODE_PINCEPLES/FRONTEND_PRICEPLES)
- [Blockchain Contract Principles](./CODE_PINCEPLES/BLOCK_CHAIN_CONTRACT_PRICEPLES)
- [Check Rules](./CODE_PINCEPLES/CHECK_RULE.md)

Key rules:
- All text in code (comments, logs, variable names, etc.) must use English uniformly, except for i18n text
- Keep functions under 20 lines, single responsibility
- No swallowing exceptions, no hidden shared state
- No deep nesting, use guard clauses

## Project Structure

See [README.md](./README.md) for top-level directory layout and frontend apps.
See [server/README.md](./server/README.md) for backend microservice domains, port allocation, and service descriptions.

## Adding New Features

After adding a backend service or feature, consider whether a corresponding admin page needs to be added to:
- [Backstage Admin](apps/backstage-admin/)
- [Customer Client App](apps/client/)

## Adding a New Microservice

See [server/README.md](./server/README.md#添加新服务清单) for the full 11-step checklist (proto → gen → impl → register → deploy).

## Microservice Required Configuration

See [server/README.md](./server/README.md#微服务必填配置清单) for placeholders that must be replaced before starting a service (payment keys, wallet RPC, ClickHouse URL, etc.).

## Database

- **All services must use PostgreSQL** (configured in `common` module, no per-service datasource needed unless multi-datasource)
- `data-pipeline` uses ClickHouse as analytics database
- Table schemas in `src/main/resources/db/schema.sql`
- Migration files in `src/main/resources/db/migration/`
- **ID strategy**: Use snowflake algorithm (`IdType.ASSIGN_ID`), NEVER use `IdType.AUTO` (auto-increment)
