import '@testing-library/jest-native/extend-expect'

global.fetch = jest.fn(() =>
  Promise.resolve({
    ok: true,
    json: () => Promise.resolve({ code: '0000', data: null }),
  })
)

jest.mock('@react-native-async-storage/async-storage', () => {
  const memory = new Map()
  const storage = {
    getItem: jest.fn((key) => Promise.resolve(memory.has(key) ? memory.get(key) : null)),
    setItem: jest.fn((key, value) => {
      memory.set(key, value)
      return Promise.resolve()
    }),
    removeItem: jest.fn((key) => {
      memory.delete(key)
      return Promise.resolve()
    }),
    clear: jest.fn(() => {
      memory.clear()
      return Promise.resolve()
    }),
    getAllKeys: jest.fn(() => Promise.resolve([...memory.keys()])),
    multiGet: jest.fn((keys) => Promise.resolve(keys.map((key) => [key, memory.has(key) ? memory.get(key) : null]))),
    multiSet: jest.fn((entries) => {
      entries.forEach(([key, value]) => memory.set(key, value))
      return Promise.resolve()
    }),
    multiRemove: jest.fn((keys) => {
      keys.forEach((key) => memory.delete(key))
      return Promise.resolve()
    }),
  }
  return { __esModule: true, default: storage }
})

jest.mock('expo-image', () => ({
  Image: 'Image',
}))

jest.mock('expo-router', () => ({
  useRouter: () => ({
    push: jest.fn(),
    back: jest.fn(),
    replace: jest.fn(),
  }),
  useLocalSearchParams: () => ({}),
  Link: 'Link',
  Stack: {
    Screen: 'Screen',
  },
}))

jest.mock('@tanstack/react-query', () => ({
  useQuery: jest.fn(),
  useMutation: jest.fn(),
  QueryClient: jest.fn().mockImplementation(() => ({
    setDefaultOptions: jest.fn(),
    getDefaultOptions: jest.fn(),
  })),
  QueryClientProvider: ({ children }) => children,
  useQueryClient: jest.fn(),
}))

Object.defineProperty(globalThis, 'performance', {
  configurable: true,
  writable: true,
  value: {
    now: jest.fn(() => Date.now()),
  },
})

beforeEach(() => {
  jest.clearAllMocks()
})