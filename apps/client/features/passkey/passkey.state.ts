import type { PasskeyHookResult, PasskeyState } from '@/features/passkey/passkey.types'

export function createIdlePasskeyState(): PasskeyState {
  return { status: 'idle', errorMessage: null, token: null }
}

export function createLoadingPasskeyState(): PasskeyState {
  return { status: 'loading', errorMessage: null, token: null }
}

export function createSuccessPasskeyState(token: string | null = null): PasskeyState {
  return { status: 'success', errorMessage: null, token }
}

export function createErrorPasskeyState(errorMessage: string): PasskeyState {
  return { status: 'error', errorMessage, token: null }
}

export function createDisabledPasskeyResult(
  noop: PasskeyHookResult['registerPasskey'],
): PasskeyHookResult {
  return {
    status: 'disabled',
    errorMessage: null,
    token: null,
    registerPasskey: noop,
    loginWithPasskey: async () => {},
    reset: () => {},
  }
}
