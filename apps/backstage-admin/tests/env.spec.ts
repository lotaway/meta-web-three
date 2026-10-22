import { describe, expect, it } from 'vitest'

import {
  API_DOC_HOST,
  BASE_SERVER_URL,
  BLOCKCHAIN_ENABLED,
  DEFAULT_BASE_SERVER_URL,
  DEFAULT_UPLOAD_URL,
  UPLOAD_URL,
} from '@/utils/env'

describe('env defaults', () => {
  it('falls back to the local gateway when no base url is configured', () => {
    expect(BASE_SERVER_URL).toBe(DEFAULT_BASE_SERVER_URL)
    expect(BASE_SERVER_URL).toBe('http://localhost:10081')
  })

  it('falls back to the platform-service minio endpoint when no upload url is configured', () => {
    expect(UPLOAD_URL).toBe(DEFAULT_UPLOAD_URL)
    expect(UPLOAD_URL).toBe('/platform-service/minio/upload')
  })

  it('falls back to the gateway base url when no api doc host is configured', () => {
    expect(API_DOC_HOST).toBe(BASE_SERVER_URL)
  })

  it('keeps the blockchain feature disabled by default', () => {
    expect(BLOCKCHAIN_ENABLED).toBe(false)
  })

  it('always resolves to a non empty base url so axios never builds empty urls', () => {
    expect(BASE_SERVER_URL.length).toBeGreaterThan(0)
    expect(API_DOC_HOST.length).toBeGreaterThan(0)
    expect(UPLOAD_URL.length).toBeGreaterThan(0)
  })
})
