import { Platform } from 'react-native'
import { getApiBaseUrl, getDevHost, DEFAULT_API_PORT } from '../../../../src/lib/api/baseUrl'

type MockState = {
  hostUri?: string
  os: 'ios' | 'android'
}

// Read lazily from globalThis so the mock factory is not affected by the
// hoisting of `jest.mock` above module-level constants.
const state = (): MockState =>
  ((globalThis as any).__baseUrlMockState ??= { hostUri: undefined, os: 'ios' }) as MockState

jest.mock('expo-constants', () => ({
  __esModule: true,
  default: {
    get expoConfig() {
      return { hostUri: state().hostUri }
    },
  },
}))

// `Platform.OS` is a value on the RN mock. Shadow it with an own property so
// both platform fallbacks can be exercised without replacing react-native
// (requireActual would bypass the preset's TurboModule mocks).
const originalOS = Platform.OS
const setOS = (os: MockState['os']) => {
  Object.defineProperty(Platform, 'OS', {
    configurable: true,
    writable: true,
    value: os,
  })
}

const ENV_KEYS = ['EXPO_PUBLIC_BACK_API_HOST', 'NEXT_PUBLIC_BACK_API_HOST'] as const
const originalEnv: Record<string, string | undefined> = {}

beforeAll(() => {
  for (const key of ENV_KEYS) originalEnv[key] = process.env[key]
})

afterAll(() => {
  for (const key of ENV_KEYS) {
    if (originalEnv[key] === undefined) delete process.env[key]
    else process.env[key] = originalEnv[key]
  }
  setOS(originalOS as MockState['os'])
  delete (globalThis as any).__baseUrlMockState
})

beforeEach(() => {
  for (const key of ENV_KEYS) delete process.env[key]
  state().hostUri = undefined
  setOS('ios')
})

describe('api/baseUrl', () => {
  describe('explicit host configuration', () => {
    it('prefers EXPO_PUBLIC_BACK_API_HOST over NEXT_PUBLIC_BACK_API_HOST', () => {
      process.env.EXPO_PUBLIC_BACK_API_HOST = 'http://expo.example.com'
      process.env.NEXT_PUBLIC_BACK_API_HOST = 'http://next.example.com'

      expect(getApiBaseUrl()).toBe('http://expo.example.com')
    })

    it('falls back to NEXT_PUBLIC_BACK_API_HOST when the EXPO value is blank', () => {
      process.env.EXPO_PUBLIC_BACK_API_HOST = '   '
      process.env.NEXT_PUBLIC_BACK_API_HOST = 'http://next.example.com'

      expect(getApiBaseUrl()).toBe('http://next.example.com')
    })

    it('trims whitespace and strips trailing slashes', () => {
      process.env.EXPO_PUBLIC_BACK_API_HOST = '  https://api.example.com///  '

      expect(getApiBaseUrl()).toBe('https://api.example.com')
    })

    it('treats empty strings as not configured', () => {
      process.env.EXPO_PUBLIC_BACK_API_HOST = ''
      process.env.NEXT_PUBLIC_BACK_API_HOST = ''

      expect(getApiBaseUrl()).toBe('http://localhost:10081')
    })
  })

  describe('dev host derivation', () => {
    it('uses the host part of Constants.expoConfig.hostUri', () => {
      state().hostUri = '192.168.1.5:8081'

      expect(getDevHost()).toBe('192.168.1.5')
      expect(getApiBaseUrl()).toBe('http://192.168.1.5:10081')
    })

    it('falls back to 10.0.2.2 on Android when hostUri is absent', () => {
      setOS('android')

      expect(getDevHost()).toBe('10.0.2.2')
      expect(getApiBaseUrl()).toBe('http://10.0.2.2:10081')
    })

    it('falls back to localhost on iOS when hostUri is absent', () => {
      setOS('ios')

      expect(getDevHost()).toBe('localhost')
      expect(getApiBaseUrl()).toBe('http://localhost:10081')
    })

    it('honours an explicit port argument', () => {
      expect(getApiBaseUrl(9999)).toBe('http://localhost:9999')
      expect(DEFAULT_API_PORT).toBe(10081)
    })
  })
})
