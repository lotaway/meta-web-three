export type PasskeyStatus = 'idle' | 'loading' | 'success' | 'error' | 'disabled'

export interface PasskeyState {
  status: PasskeyStatus
  errorMessage: string | null
  token: string | null
}

export interface PasskeyHookResult extends PasskeyState {
  registerPasskey: (userName: string) => Promise<void>
  loginWithPasskey: () => Promise<void>
  reset: () => void
}
