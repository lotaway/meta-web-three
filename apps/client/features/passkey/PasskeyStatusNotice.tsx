import { Text, TouchableOpacity, View } from 'react-native'
import { IconSymbol } from '@/components/ui/IconSymbol'
import { PASSKEY_MESSAGES } from '@/features/passkey/passkey.messages'
import { passkeyStyles } from '@/features/passkey/passkey.styles'
import type { PasskeyStatus } from '@/features/passkey/passkey.types'

type PasskeyStatusNoticeProps = {
  status: PasskeyStatus
  token: string | null
  errorMessage: string | null
  onReset: () => void
}

export function PasskeyStatusNotice({
  status,
  token,
  errorMessage,
  onReset,
}: PasskeyStatusNoticeProps) {
  if (status === 'success') {
    return (
      <View style={[passkeyStyles.statusBox, { backgroundColor: '#f0fff4' }]}>
        <IconSymbol name="checkmark.circle.fill" size={16} color="#4cd964" />
        <Text style={[passkeyStyles.statusText, { color: '#4cd964' }]}>
          {token ? PASSKEY_MESSAGES.loginSuccess : PASSKEY_MESSAGES.registerSuccess}
        </Text>
      </View>
    )
  }

  if (status !== 'error' || !errorMessage) {
    return null
  }

  return (
    <View style={[passkeyStyles.statusBox, { backgroundColor: '#fff2f2' }]}>
      <IconSymbol name="exclamationmark.triangle.fill" size={16} color="#dd524d" />
      <Text style={[passkeyStyles.statusText, { color: '#dd524d' }]}>{errorMessage}</Text>
      <TouchableOpacity onPress={onReset} style={passkeyStyles.retryButton}>
        <Text style={passkeyStyles.retryText}>{PASSKEY_MESSAGES.retryButton}</Text>
      </TouchableOpacity>
    </View>
  )
}
