import '@testing-library/jest-native/extend-expect'

global.fetch = jest.fn(() =>
  Promise.resolve({
    ok: true,
    json: () => Promise.resolve({ code: '0000', data: null }),
  })
)

jest.mock('@react-native-async-storage/async-storage', () =>
  require('@react-native-async-storage/async-storage/jest/async-storage-mock')
)

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