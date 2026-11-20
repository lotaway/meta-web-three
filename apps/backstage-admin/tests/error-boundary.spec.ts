import { describe, expect, it } from 'vitest'
import { normalizeError } from '@/utils/errorBoundary'

describe('normalizeError', () => {
  it('uses the message of an Error', () => {
    expect(normalizeError(new Error('network down'))).toEqual({
      message: 'network down',
    })
  })

  it('falls back to the error name when the message is empty', () => {
    expect(normalizeError(new Error(''))).toEqual({ message: 'Error' })
  })

  it('keeps a non-empty thrown string', () => {
    expect(normalizeError('timeout')).toEqual({ message: 'timeout' })
  })

  it('returns a generic message for unknown values', () => {
    expect(normalizeError(null)).toEqual({ message: 'Unexpected error' })
    expect(normalizeError({ code: 500 })).toEqual({
      message: 'Unexpected error',
    })
  })
})
