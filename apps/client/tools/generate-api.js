const { spawnSync } = require('child_process')
const path = require('path')

const clientDir = path.resolve(__dirname, '..')

const env = {
  ...process.env,
  NODE_PATH: process.env.NODE_PATH || path.resolve(clientDir, 'node_modules'),
  OPENAPI_ENV_FILES: '.env.development,.env.local',
  OPENAPI_API_HOST_ENV: 'NEXT_PUBLIC_BACK_API_HOST',
  OPENAPI_API_DOC_HOST_ENV: 'NEXT_PUBLIC_BACK_API_DOC_HOST',
  OPENAPI_RUNTIME_ENV_VAR: 'process.env.NEXT_PUBLIC_BACK_API_HOST',
  OPENAPI_TIMEOUT: '120000',
}

function run(script, args) {
  const result = spawnSync('node', [script, ...args], {
    stdio: 'inherit',
    env,
    cwd: clientDir,
  })
  if (result.status !== 0) {
    process.exit(result.status)
  }
}

// Step 1: Download OpenAPI docs + generate TypeScript
run(path.resolve(__dirname, '../../tools/OpenApiToTS.js'), [])

// Step 2: Download GraphQL schema
env.SCHEMA_ENV_FILES = '.env.development,.env.local'
env.SCHEMA_API_HOST_ENV = 'NEXT_PUBLIC_BACK_API_HOST'
run(path.resolve(__dirname, '../../tools/DownloadGraphQLSchema.js'), [])

// Step 3: Generate TypeScript from GraphQL
const codegenResult = spawnSync('npx', ['graphql-codegen', '--config', 'codegen.ts'], {
  stdio: 'inherit',
  env,
  cwd: clientDir,
  shell: true,
})
if (codegenResult.status !== 0) {
  process.exit(codegenResult.status)
}
