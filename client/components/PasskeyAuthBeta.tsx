import { useEffect } from 'react'
import { Text, View } from 'react-native'
import { PasskeyActionButton } from '@/features/passkey/PasskeyActionButton'
import { PASSKEY_MESSAGES } from '@/features/passkey/passkey.messages'
import { passkeyStyles } from '@/features/passkey/passkey.styles'
import { usePasskey } from '@/hooks/usePasskey'
import { FEATURE_PASSKEY_ENABLED } from '@/constants/Features'

type Props = {
  label?: string
  onAuthorized?: (token: string | null) => void
  onError?: (message: string) => void
}

export default function PasskeyAuthBeta({
  label = PASSKEY_MESSAGES.betaDefaultLabel,
  onAuthorized,
  onError,
}: Props) {
  const { status, errorMessage, token, loginWithPasskey, reset } = usePasskey()

  useEffect(() => {
    if (status === 'success') {
      onAuthorized?.(token)
    }

    if (status === 'error' && errorMessage) {
      onError?.(errorMessage)
    }

    if (status === 'success' || status === 'error') {
      const timer = setTimeout(reset, 2500)
      return () => clearTimeout(timer)
    }
  }, [errorMessage, onAuthorized, onError, reset, status, token])

  if (!FEATURE_PASSKEY_ENABLED) {
    return null
  }

  return (
    <View>
      <PasskeyActionButton
        label={label}
        onPress={loginWithPasskey}
        status={status}
        idleIconName="faceid"
        successLabel={PASSKEY_MESSAGES.betaSuccessLabel}
        errorLabel={PASSKEY_MESSAGES.betaRetryLabel}
        idleBackgroundColor="#fa436a"
      />
      {status === 'error' && errorMessage ? (
        <Text style={passkeyStyles.errorHint}>{errorMessage}</Text>
      ) : null}
    </View>
  )
}
