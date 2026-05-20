import { PASSKEY_MESSAGES } from '@/features/passkey/passkey.messages'

type PasskeyErrorScope = 'register' | 'login'

type PasskeyRuntimeError = {
  code?: string
  message?: string
}

export function getPasskeyErrorMessage(
  error: unknown,
  scope: PasskeyErrorScope,
): string {
  const runtimeError = error as PasskeyRuntimeError

  if (runtimeError?.code === 'BUSY') {
    return PASSKEY_MESSAGES.busy
  }

  if (runtimeError?.code === 'UNSUPPORTED') {
    return PASSKEY_MESSAGES.unsupported
  }

  if (scope === 'login' && runtimeError?.code === 'AUTH_ERROR') {
    return PASSKEY_MESSAGES.authFailed
  }

  if (runtimeError?.message) {
    return runtimeError.message
  }

  return scope === 'register'
    ? PASSKEY_MESSAGES.registerFailed
    : PASSKEY_MESSAGES.loginFailed
}
