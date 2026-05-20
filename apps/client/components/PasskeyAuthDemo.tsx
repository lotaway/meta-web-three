import { useEffect, useState } from 'react'
import {
  Alert,
  Text,
  TextInput,
  View,
} from 'react-native'
import { IconSymbol } from '@/components/ui/IconSymbol'
import { PasskeyActionButton } from '@/features/passkey/PasskeyActionButton'
import { PASSKEY_MESSAGES } from '@/features/passkey/passkey.messages'
import { passkeyStyles } from '@/features/passkey/passkey.styles'
import { PasskeyStatusNotice } from '@/features/passkey/PasskeyStatusNotice'
import { usePasskey } from '@/hooks/usePasskey'
import { FEATURE_PASSKEY_ENABLED } from '@/constants/Features'

type Props = {
  userId?: number
  userName?: string
  onLoginSuccess?: (token: string) => void
}

export default function PasskeyAuthDemo({
  userId,
  userName,
  onLoginSuccess,
}: Props) {
  const passkeyFeatureEnabled = FEATURE_PASSKEY_ENABLED
  const [inputName, setInputName] = useState(userName ?? '')
  const {
    status,
    errorMessage,
    token,
    registerPasskey,
    loginWithPasskey,
    reset,
  } = usePasskey(userId)

  useEffect(() => {
    setInputName(userName ?? '')
  }, [userName])

  async function handleRegister() {
    const trimmedName = inputName.trim()

    if (!trimmedName) {
      Alert.alert('提示', PASSKEY_MESSAGES.registerPrompt)
      return
    }

    await registerPasskey(trimmedName)
  }

  async function handleLogin() {
    await loginWithPasskey()
  }

  useEffect(() => {
    if (status !== 'success' || !token || !onLoginSuccess) {
      return
    }

    onLoginSuccess(token)
  }, [onLoginSuccess, status, token])

  if (!passkeyFeatureEnabled) {
    return null
  }

  return (
    <View style={passkeyStyles.card}>
      <View style={passkeyStyles.header}>
        <View style={passkeyStyles.iconWrap}>
          <IconSymbol name="faceid" size={22} color="#fff" />
        </View>
        <Text style={passkeyStyles.title}>{PASSKEY_MESSAGES.title}</Text>
      </View>

      <Text style={passkeyStyles.description}>{PASSKEY_MESSAGES.description}</Text>

      {userId !== undefined ? (
        <View style={passkeyStyles.section}>
          <Text style={passkeyStyles.sectionLabel}>{PASSKEY_MESSAGES.registerSection}</Text>
          <TextInput
            style={passkeyStyles.input}
            value={inputName}
            onChangeText={setInputName}
            placeholder="用于标识该凭证的用户名"
            placeholderTextColor="#b0b3ba"
            editable={status !== 'loading'}
          />
          <PasskeyActionButton
            label={PASSKEY_MESSAGES.registerButton}
            onPress={handleRegister}
            status={status}
            idleIconName="plus.circle.fill"
            successLabel={PASSKEY_MESSAGES.registerButton}
            errorLabel={PASSKEY_MESSAGES.retryButton}
            idleBackgroundColor="#4399fc"
          />
        </View>
      ) : null}

      {userId !== undefined ? <View style={passkeyStyles.divider} /> : null}

      <View style={passkeyStyles.section}>
        <Text style={passkeyStyles.sectionLabel}>{PASSKEY_MESSAGES.loginSection}</Text>
        <PasskeyActionButton
          label={PASSKEY_MESSAGES.loginButton}
          onPress={handleLogin}
          status={status}
          idleIconName="faceid"
          successLabel={PASSKEY_MESSAGES.betaSuccessLabel}
          errorLabel={PASSKEY_MESSAGES.betaRetryLabel}
          idleBackgroundColor="#fa436a"
        />
      </View>

      <PasskeyStatusNotice
        status={status}
        token={token}
        errorMessage={errorMessage}
        onReset={reset}
      />
    </View>
  )
}
