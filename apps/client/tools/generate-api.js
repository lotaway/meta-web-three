const { spawnSync } = require('child_process')
const path = require('path')

const clientDir = path.resolve(__dirname, '..')

const env = {
  ...process.env,
  NODE_PATH: process.env.NODE_PATH || path.resolve(clientDir, 'node_modules'),
  // Load .env first so NEXT_PUBLIC_BACK_API_HOST / NEXT_PUBLIC_BACK_API_DOC_HOST
  // are honoured; .env.development and .env.local may override them.
  OPENAPI_ENV_FILES: '.env,.env.development,.env.local',
  OPENAPI_API_HOST_ENV: 'NEXT_PUBLIC_BACK_API_HOST',
  OPENAPI_API_DOC_HOST_ENV: 'NEXT_PUBLIC_BACK_API_DOC_HOST',
  OPENAPI_RUNTIME_ENV_VAR: 'process.env.NEXT_PUBLIC_BACK_API_HOST',
  OPENAPI_TIMEOUT: '120000',
}

function run(script, args, { allowFailure = false } = {}) {
  const result = spawnSync('node', [script, ...args], {
    stdio: 'inherit',
    env,
    cwd: clientDir,
  })
  if (result.status !== 0) {
    if (allowFailure) {
      console.warn(
        `[generate:api] ${path.basename(script)} failed with exit code ${result.status}; ` +
          'continuing without GraphQL output.'
      )
      return false
    }
    process.exit(result.status)
  }
  return true
}

// Step 1: Download OpenAPI docs + generate TypeScript (must succeed).
run(path.resolve(__dirname, '../../tools/OpenApiToTS.js'), [])

// Step 2: Download GraphQL schema. This needs a running gateway, so failure
// must not block the REST client generated in step 1.
env.SCHEMA_ENV_FILES = '.env,.env.development,.env.local'
env.SCHEMA_API_HOST_ENV = 'NEXT_PUBLIC_BACK_API_HOST'
const schemaDownloaded = run(
  path.resolve(__dirname, '../../tools/DownloadGraphQLSchema.js'),
  [],
  { allowFailure: true }
)

// Step 3: Generate TypeScript from GraphQL only when a schema is available.
if (schemaDownloaded) {
  const codegenResult = spawnSync('npx', ['graphql-codegen', '--config', 'codegen.ts'], {
    stdio: 'inherit',
    env,
    cwd: clientDir,
    shell: true,
  })
  if (codegenResult.status !== 0) {
    console.warn(
      `[generate:api] graphql-codegen failed with exit code ${codegenResult.status}; ` +
        'REST client generation already completed successfully.'
    )
  }
} else {
  console.warn('[generate:api] Skipping graphql-codegen because no schema was downloaded.')
}
