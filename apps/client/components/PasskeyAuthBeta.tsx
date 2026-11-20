import { useEffect } from 'react'
import { Text, View } from 'react-native'
import { PasskeyActionButton } from '@/features/passkey/PasskeyActionButton'
import { PASSKEY_MESSAGES } from '@/features/passkey/passkey.messages'
import { passkeyStyles } from '@/features/passkey/passkey.styles'
import { usePasskey } from '@/hooks/usePasskey'
import type { PasskeyStatus } from '@/features/passkey/passkey.types'
import { FEATURE_PASSKEY_ENABLED } from '@/constants/Features'

type Props = {
  label?: string
  onAuthorized?: (token: string | null) => void
  onError?: (message: string) => void
}

function applyPasskeyStateChange(
  status: PasskeyStatus,
  token: string | null,
  errorMessage: string | null,
  onAuthorized: ((token: string | null) => void) | undefined,
  onError: ((message: string) => void) | undefined,
  reset: () => void,
) {
  if (status === 'success') onAuthorized?.(token)

  if (status === 'error' && errorMessage) onError?.(errorMessage)

  if (status === 'success' || status === 'error') {
    const timer = setTimeout(reset, 2500)
    return () => clearTimeout(timer)
  }
}

function PasskeyBetaAction({ label, status, onPress }: {
  label: string
  status: PasskeyStatus
  onPress: () => void
}) {
  return (
    <PasskeyActionButton
      label={label}
      onPress={onPress}
      status={status}
      idleIconName="faceid"
      successLabel={PASSKEY_MESSAGES.betaSuccessLabel}
      errorLabel={PASSKEY_MESSAGES.betaRetryLabel}
      idleBackgroundColor="#fa436a"
    />
  )
}

export default function PasskeyAuthBeta({
  label = PASSKEY_MESSAGES.betaDefaultLabel,
  onAuthorized,
  onError,
}: Props) {
  const { status, errorMessage, token, loginWithPasskey, reset } = usePasskey()

  useEffect(() => {
    return applyPasskeyStateChange(status, token, errorMessage, onAuthorized, onError, reset)
  }, [errorMessage, onAuthorized, onError, reset, status, token])

  if (!FEATURE_PASSKEY_ENABLED) {
    return null
  }

  return (
    <View>
      <PasskeyBetaAction label={label} status={status} onPress={loginWithPasskey} />
      {status === 'error' && errorMessage ? (
        <Text style={passkeyStyles.errorHint}>{errorMessage}</Text>
      ) : null}
    </View>
  )
}
