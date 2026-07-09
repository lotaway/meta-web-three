const { spawnSync } = require('child_process')
const path = require('path')

const env = {
  ...process.env,
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
    cwd: __dirname,
  })
  if (result.status !== 0) {
    process.exit(result.status)
  }
}

run(path.resolve(__dirname, '../../tools/OpenApiToTS.js'), [])

env.SCHEMA_ENV_FILES = '.env.development,.env.local'
env.SCHEMA_API_HOST_ENV = 'NEXT_PUBLIC_BACK_API_HOST'
run(path.resolve(__dirname, '../../tools/DownloadGraphQLSchema.js'), [])

const codegenPath = path.resolve(__dirname, '../node_modules/@graphql-codegen/cli/cjs/bin.js')
run(codegenPath, ['--config', 'codegen.ts'])
