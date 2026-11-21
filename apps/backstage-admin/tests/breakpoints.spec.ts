import { describe, expect, it } from 'vitest'
import { resolveDevice } from '@/views/layout/composables/useResizeHandler'

describe('responsive breakpoints', () => {
  it('treats widths below 640 as compact', () => {
    expect(resolveDevice(0)).toBe('compact')
    expect(resolveDevice(639)).toBe('compact')
  })

  it('treats widths from 640 to 1023 as medium', () => {
    expect(resolveDevice(640)).toBe('medium')
    expect(resolveDevice(1023)).toBe('medium')
  })

  it('treats widths from 1024 up as expanded', () => {
    expect(resolveDevice(1024)).toBe('expanded')
    expect(resolveDevice(2560)).toBe('expanded')
  })
})
