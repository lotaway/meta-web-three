import { ActivityIndicator, Text, TouchableOpacity } from 'react-native'
import { IconSymbol } from '@/components/ui/IconSymbol'
import { passkeyStyles } from '@/features/passkey/passkey.styles'
import type { PasskeyStatus } from '@/features/passkey/passkey.types'

type PasskeyActionButtonProps = {
  label: string
  onPress: () => void
  status: PasskeyStatus
  idleIconName: string
  successLabel?: string
  errorLabel?: string
  idleBackgroundColor: string
}

function getButtonStyle(
  status: PasskeyStatus,
  idleBackgroundColor: string,
) {
  if (status === 'success') {
    return { backgroundColor: '#4cd964' }
  }

  if (status === 'error') {
    return { backgroundColor: '#dd524d' }
  }

  if (status === 'loading') {
    return [passkeyStyles.buttonDisabled, { backgroundColor: idleBackgroundColor }]
  }

  return { backgroundColor: idleBackgroundColor }
}

function getButtonContent(
  status: PasskeyStatus,
  label: string,
  idleIconName: string,
  successLabel: string,
  errorLabel: string,
) {
  if (status === 'loading') {
    return <ActivityIndicator color="#fff" size="small" />
  }

  if (status === 'success') {
    return (
      <>
        <IconSymbol name="checkmark.shield.fill" size={18} color="#fff" />
        <Text style={passkeyStyles.buttonText}>{successLabel}</Text>
      </>
    )
  }

  if (status === 'error') {
    return (
      <>
        <IconSymbol name="exclamationmark.triangle.fill" size={18} color="#fff" />
        <Text style={passkeyStyles.buttonText}>{errorLabel}</Text>
      </>
    )
  }

  return (
    <>
      <IconSymbol name={idleIconName as never} size={18} color="#fff" />
      <Text style={passkeyStyles.buttonText}>{label}</Text>
    </>
  )
}

export function PasskeyActionButton({
  label,
  onPress,
  status,
  idleIconName,
  successLabel = label,
  errorLabel = label,
  idleBackgroundColor,
}: PasskeyActionButtonProps) {
  return (
    <TouchableOpacity
      style={[passkeyStyles.button, getButtonStyle(status, idleBackgroundColor)]}
      onPress={onPress}
      disabled={status === 'loading'}
      activeOpacity={0.82}>
      {getButtonContent(status, label, idleIconName, successLabel, errorLabel)}
    </TouchableOpacity>
  )
}
