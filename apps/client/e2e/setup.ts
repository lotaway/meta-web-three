import { cleanup, init, reload } from 'detox'
import adapter from 'detox-adapter-jest'
import { configureByJSON, matchers } from 'detox-matchers'

jest.setTimeout(120000)
jest.mock('detox', () => ({
  ...jest.requireActual('detox'),
}))

expect.extend(matchers)

beforeAll(async () => {
  await adapter.beforeAll()
})

beforeEach(async () => {
  await adapter.beforeEach()
})

afterAll(async () => {
  await adapter.afterAll()
})

afterEach(async () => {
  await cleanup()
})

global.describe = global.describe.skip

export { init, cleanup, reload, adapter }
export * from 'detox-matchers'