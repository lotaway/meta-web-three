const { spawnSync } = require('child_process')
const path = require('path')

const env = {
  ...process.env,
  OPENAPI_ENV_FILES: '.env,.env.local',
  OPENAPI_API_HOST_ENV: 'VITE_BASE_SERVER_URL',
  OPENAPI_API_DOC_HOST_ENV: 'VITE_API_DOC_HOST',
  OPENAPI_RUNTIME_ENV_VAR: 'import.meta.env.VITE_BASE_SERVER_URL',
  OPENAPI_TIMEOUT: '180000',
  OPENAPI_DEFAULT_HOST: 'http://localhost:10081',
}

const result = spawnSync('node', [
  path.resolve(__dirname, '../../tools/OpenApiToTS.js'),
], {
  stdio: 'inherit',
  env,
  cwd: path.resolve(__dirname, '..'),
})
if (result.status !== 0) {
  process.exit(result.status)
}
