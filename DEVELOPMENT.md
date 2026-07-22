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

### 3. Generate Frontend API Client

After backend controllers are deployed, generate TypeScript clients from OpenAPI docs.

**Prerequisites:**
```bash
npm install -g @openapitools/openapi-generator-cli
```

**For [backstage-admin](apps/backstage-admin/) and [customer client](apps/client/):**
Both use the shared script `apps/tools/OpenApiToTS.js`.
Configure `.env.development`:
```bash
NEXT_PUBLIC_BACK_API_HOST=http://localhost:10081
NEXT_PUBLIC_BACK_API_DOC_HOST=http://localhost:10081
```
Then run in the app directory:
```bash
yarn generate:api
```
Generated code goes to `src/generated/api/`. See full details in [apps/client/README.md](./apps/client/README.md#generate-api-interfaces).

**For [digital-twin](apps/digital-twin/):**
⚠️ `generate:api` is NOT yet set up. API types in `src/renderer/services/generated/` are currently written manually. When adding new backend endpoints, manually create corresponding types and service calls in `src/renderer/services/api/`.

### 4. Frontend Pages
- [Backstage Admin](apps/backstage-admin/) — enterprise admin panel (Vue 3 + Element Plus)
- [Customer Client App](apps/client/) — customer mobile app (React Native / Expo)
- [Digital Twin](apps/digital-twin/) — factory digital twin desktop (Electron + React)
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
- [Digital Twin](apps/digital-twin/)

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
