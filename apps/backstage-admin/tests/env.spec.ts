import { describe, expect, it } from 'vitest'

import {
  API_DOC_HOST,
  BASE_SERVER_URL,
  BLOCKCHAIN_ENABLED,
  DEFAULT_BASE_SERVER_URL,
  DEFAULT_UPLOAD_URL,
  UPLOAD_URL,
  resolveEnv,
} from '@/utils/env'

describe('env resolution', () => {
  it('falls back to the local gateway when no base url is configured', () => {
    const env = resolveEnv({})
    expect(env.BASE_SERVER_URL).toBe(DEFAULT_BASE_SERVER_URL)
    expect(env.BASE_SERVER_URL).toBe('http://localhost:10081')
  })

  it('falls back to the platform-service minio endpoint when no upload url is configured', () => {
    const env = resolveEnv({})
    expect(env.UPLOAD_URL).toBe(DEFAULT_UPLOAD_URL)
    expect(env.UPLOAD_URL).toBe('/platform-service/minio/upload')
  })

  it('falls back to the gateway base url when no api doc host is configured', () => {
    const env = resolveEnv({})
    expect(env.API_DOC_HOST).toBe(env.BASE_SERVER_URL)
  })

  it('keeps the blockchain feature disabled by default', () => {
    expect(resolveEnv({}).BLOCKCHAIN_ENABLED).toBe(false)
  })

  it('always resolves to non empty urls so axios never builds empty urls', () => {
    const env = resolveEnv({})
    expect(env.BASE_SERVER_URL.length).toBeGreaterThan(0)
    expect(env.API_DOC_HOST.length).toBeGreaterThan(0)
    expect(env.UPLOAD_URL.length).toBeGreaterThan(0)
  })

  it('honors explicit variables regardless of the ambient .env file', () => {
    const env = resolveEnv({
      VITE_BASE_SERVER_URL: 'http://10.0.0.5:10081',
      VITE_API_DOC_HOST: 'http://docs.internal:18081',
      VITE_UPLOAD_URL: '/custom/upload',
      VITE_BLOCKCHAIN_ENABLED: 'TRUE',
    })
    expect(env.BASE_SERVER_URL).toBe('http://10.0.0.5:10081')
    expect(env.API_DOC_HOST).toBe('http://docs.internal:18081')
    expect(env.UPLOAD_URL).toBe('/custom/upload')
    expect(env.BLOCKCHAIN_ENABLED).toBe(true)
  })

  it('trims blank values back to defaults', () => {
    const env = resolveEnv({ VITE_BASE_SERVER_URL: '   ', VITE_BLOCKCHAIN_ENABLED: '' })
    expect(env.BASE_SERVER_URL).toBe(DEFAULT_BASE_SERVER_URL)
    expect(env.BLOCKCHAIN_ENABLED).toBe(false)
  })

  it('exports constants resolved from the active Vite env', () => {
    const expected = resolveEnv(import.meta.env)
    expect(BASE_SERVER_URL).toBe(expected.BASE_SERVER_URL)
    expect(API_DOC_HOST).toBe(expected.API_DOC_HOST)
    expect(UPLOAD_URL).toBe(expected.UPLOAD_URL)
    expect(BLOCKCHAIN_ENABLED).toBe(expected.BLOCKCHAIN_ENABLED)
  })
})
