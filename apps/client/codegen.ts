import type { CodegenConfig } from '@graphql-codegen/cli'

const config: CodegenConfig = {
  schema: './graphql/schema.graphql',
  generates: {
    './src/generated/graphql/types.ts': {
      plugins: ['typescript', 'typescript-operations'],
      config: {
        skipTypename: true,
        onlyOperationTypes: false,
      },
    },
  },
}

export default config
