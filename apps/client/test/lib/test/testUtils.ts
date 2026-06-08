import { renderHook, act, waitFor } from '@testing-library/react-native'

export { act, waitFor }

export function createMockApiResponse<T>(data: T, code: string = '0000', message: string = 'success') {
  return {
    code,
    message,
    data,
    timestamp: Date.now(),
  }
}

export function createMockErrorResponse(message: string, code: string = '9999') {
  return {
    code,
    message,
    data: null,
    timestamp: Date.now(),
  }
}

export function mockFetch(data: any, ok: boolean = true) {
  return jest.fn().mockResolvedValue({
    ok,
    json: () => Promise.resolve(data),
    status: ok ? 200 : 400,
    headers: new Headers({ 'content-type': 'application/json' }),
  })
}

export function setupQueryClient(overrides: any = {}) {
  return {
    setDefaultOptions: jest.fn(),
    getDefaultOptions: jest.fn(),
    ...overrides,
  }
}

export * from '@testing-library/react-native'