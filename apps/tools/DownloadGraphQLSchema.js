// GraphQL Schema downloader for client code generation
// Downloads the GraphQL schema via introspection from the gateway
//
// Config via environment variables:
//   SCHEMA_ENV_FILES          - comma-separated env files (default: .env)
//   SCHEMA_API_HOST_ENV       - env var name for API host (default: VITE_BASE_SERVER_URL)
//   SCHEMA_OUTPUT_DIR         - output dir relative to cwd (default: graphql)
//   SCHEMA_DEFAULT_HOST       - fallback host (default: http://localhost:10081)

const fs = require('node:fs')
const https = require('node:https')
const http = require('node:http')
const path = require('node:path')
const dotenv = require('dotenv')

const projectRoot = process.cwd()

function readConfig() {
  const envFiles = (process.env.SCHEMA_ENV_FILES || '.env').split(',')
  for (const file of envFiles) {
    const filePath = path.resolve(projectRoot, file.trim())
    if (fs.existsSync(filePath)) {
      dotenv.config({ path: filePath })
    }
  }

  const apiHostEnv = process.env.SCHEMA_API_HOST_ENV || 'VITE_BASE_SERVER_URL'
  const defaultHost = process.env.SCHEMA_DEFAULT_HOST || 'http://localhost:10081'

  return {
    apiHost: process.env[apiHostEnv] || defaultHost,
    outputDir: process.env.SCHEMA_OUTPUT_DIR || 'graphql',
  }
}

function getIntrospectionUrl(config) {
  return `${config.apiHost}/graphql`
}

async function downloadSchema(config) {
  const url = getIntrospectionUrl(config)
  const outputDir = path.resolve(projectRoot, config.outputDir)
  const outputPath = path.resolve(outputDir, 'schema.graphql')

  await fs.promises.mkdir(outputDir, { recursive: true })

  const introspectionQuery = JSON.stringify({
    query: `
      query IntrospectionQuery {
        __schema {
          queryType { name }
          mutationType { name }
          subscriptionType { name }
          types {
            ...FullType
          }
          directives {
            name
            description
            locations
            args {
              ...InputValue
            }
          }
        }
      }
      fragment FullType on __Type {
        kind
        name
        description
        fields(includeDeprecated: true) {
          name
          description
          args {
            ...InputValue
          }
          type {
            ...TypeRef
          }
          isDeprecated
          deprecationReason
        }
        inputFields {
          ...InputValue
        }
        interfaces {
          ...TypeRef
        }
        enumValues(includeDeprecated: true) {
          name
          description
          isDeprecated
          deprecationReason
        }
        possibleTypes {
          ...TypeRef
        }
      }
      fragment InputValue on __InputValue {
        name
        description
        type { ...TypeRef }
        defaultValue
      }
      fragment TypeRef on __Type {
        kind
        name
        ofType {
          kind
          name
          ofType {
            kind
            name
            ofType {
              kind
              name
              ofType {
                kind
                name
                ofType {
                  kind
                  name
                  ofType {
                    kind
                    name
                    ofType {
                      kind
                      name
                    }
                  }
                }
              }
            }
          }
        }
      }
    `,
  })

  return new Promise((resolve, reject) => {
    const urlObj = new URL(url)
    const options = {
      hostname: urlObj.hostname,
      port: urlObj.port,
      path: urlObj.pathname,
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Content-Length': Buffer.byteLength(introspectionQuery),
      },
    }

    const client = urlObj.protocol === 'https:' ? https : http
    const req = client.request(options, (res) => {
      let data = ''
      res.on('data', (chunk) => { data += chunk })
      res.on('end', async () => {
        try {
          const result = JSON.parse(data)
          if (result.errors) {
            console.warn('Introspection errors:', result.errors)
          }
          // Build the SDL string from the introspection result
          const { buildClientSchema, printSchema } = require('graphql')
          const schema = buildClientSchema(result.data)
          const sdl = printSchema(schema)
          await fs.promises.writeFile(outputPath, sdl, 'utf-8')
          console.log(`GraphQL schema saved to ${outputPath}`)
          resolve(outputPath)
        } catch (err) {
          reject(new Error(`Failed to process schema: ${err.message}`))
        }
      })
    })

    req.on('error', (err) => reject(err))
    req.write(introspectionQuery)
    req.end()
  })
}

async function main() {
  const config = readConfig()
  try {
    console.log('Downloading GraphQL schema from', getIntrospectionUrl(config))
    await downloadSchema(config)
    console.log('Schema download complete.')
  } catch (error) {
    console.error('Failed to download GraphQL schema:', error.message)
    process.exit(1)
  }
}

main()
