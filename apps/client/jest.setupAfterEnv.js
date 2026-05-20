import '@testing-library/jest-native/extend-expect'

global.fetch = jest.fn(() =>
  Promise.resolve({
    ok: true,
    json: () => Promise.resolve({ code: '0000', data: null }),
    headers: new Headers({ 'content-type': 'application/json' }),
    status: 200,
  })
) as jest.Mock

global.performance = {
  now: jest.fn(() => Date.now()),
  mark: jest.fn(),
  measure: jest.fn(),
  clearMarks: jest.fn(),
  clearMeasures: jest.fn(),
  getEntriesByType: jest.fn(() => []),
  getEntriesByName: jest.fn(() => []),
} as any

global.IntersectionObserver = jest.fn().mockImplementation(() => ({
  observe: jest.fn(),
  unobserve: jest.fn(),
  disconnect: jest.fn(),
  takeRecords: jest.fn(),
}))

global.ResizeObserver = jest.fn().mockImplementation(() => ({
  observe: jest.fn(),
  unobserve: jest.fn(),
  disconnect: jest.fn(),
}))

Object.defineProperty(window, 'matchMedia', {
  writable: true,
  value: jest.fn().mockImplementation((query) => ({
    matches: false,
    media: query,
    onchange: null,
    addListener: jest.fn(),
    removeListener: jest.fn(),
    addEventListener: jest.fn(),
    removeEventListener: jest.fn(),
    dispatchEvent: jest.fn(),
  })),
})

jest.mock('expo-image', () => ({
  Image: 'Image',
}))

jest.mock('expo-router', () => ({
  useRouter: () => ({
    push: jest.fn(),
    back: jest.fn(),
    replace: jest.fn(),
    prefetch: jest.fn(),
  }),
  useLocalSearchParams: () => ({}),
  useGlobalSearchParams: () => ({}),
  Link: ({ children, href }: any) => children || href,
  Href: ({ children }: any) => children,
  Stack: {
    Screen: 'Screen',
  },
  Tabs: {
    Screen: 'Screen',
  },
  useSegments: () => [],
  useRootLayout: () => ({ width: 375, height: 812 }),
}))

jest.mock('@react-native-async-storage/async-storage', () =>
  require('@react-native-async-storage/async-storage/jest/async-storage-mock')
)

jest.mock('@tanstack/react-query', () => {
  const originalModule = jest.requireActual('@tanstack/react-query')
  return {
    ...originalModule,
    useQuery: jest.fn(),
    useMutation: jest.fn(),
    useQueryClient: jest.fn(() => ({
      invalidateQueries: jest.fn(),
      setQueryData: jest.fn(),
      getQueryData: jest.fn(),
      clear: jest.fn(),
    })),
    QueryClient: jest.fn().mockImplementation(() => ({
      setDefaultOptions: jest.fn(),
      getDefaultOptions: jest.fn(),
      clear: jest.fn(),
      invalidateQueries: jest.fn(),
    })),
    useInfiniteQuery: jest.fn(),
    useQueries: jest.fn(),
  }
})

jest.mock('@stripe/stripe-react-native', () => ({
  init: jest.fn(),
  createPaymentSheet: jest.fn(() => Promise.resolve({})),
  presentPaymentSheet: jest.fn(() => Promise.resolve({})),
  retrievePaymentSheetParams: jest.fn(() => Promise.resolve({})),
}))

jest.mock('react-native-reanimated', () => {
  const Reanimated = require('react-native-reanimated/mock')
  Reanimated.default.call = () => {}
  return Reanimated
})

jest.mock('react-native-gesture-handler', () => {
  const View = require('react-native').View
  return {
    Swipeable: View,
    DrawerLayout: View,
    State: {},
    ScrollView: View,
    Slider: View,
    Switch: View,
    TextInput: View,
    ToolbarAndroid: View,
    ViewPagerAndroid: View,
    DrawerLayoutAndroid: View,
    WebView: View,
    NativeViewGestureHandler: View,
    TapGestureHandler: View,
    FlingGestureHandler: View,
    ForceTouchGestureHandler: View,
    LongPressGestureHandler: View,
    PanGestureHandler: View,
    PinchGestureHandler: View,
    RotationGestureHandler: View,
    RawButton: View,
    BaseButton: View,
    RectButton: View,
    BorderlessButton: View,
    FlatList: View,
    gestureHandlerRootHOC: jest.fn(),
    Directions: {},
    Gesture: {},
    GestureDetector: ({ children }: any) => children,
    GestureHandlerRootView: View,
  }
})

jest.mock('expo-font', () => ({
  isLoaded: jest.fn(() => true),
  loadAsync: jest.fn(() => Promise.resolve()),
}))

jest.mock('expo-constants', () => ({
  default: {
    manifest: {},
    systemFonts: [],
  },
}))

jest.mock('expo-linking', () => ({
  openURL: jest.fn(() => Promise.resolve()),
  canOpenURL: jest.fn(() => Promise.resolve(true)),
  getInitialURL: jest.fn(() => Promise.resolve(null)),
  addEventListener: jest.fn(),
  removeEventListener: jest.fn(),
}))

jest.mock('expo-localization', () => ({
  locale: 'en-US',
  locales: ['en-US'],
  timezone: 'America/New_York',
  isRTL: false,
  region: 'US',
  getCalendars: jest.fn(() => []),
  getLocales: jest.fn(() => [{ languageTag: 'en-US' }]),
}))

console.error = jest.fn()
console.warn = jest.fn()
console.log = jest.fn()