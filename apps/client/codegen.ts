import type { CodegenConfig } from '@graphql-codegen/cli'

const config: CodegenConfig = {
  schema: './graphql/schema.graphql',
  documents: './graphql/*.graphql',
  generates: {
    './src/generated/graphql/types.ts': {
      plugins: ['typescript', 'typescript-operations'],
    },
  },
}

export default config
