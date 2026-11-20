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

function createRegisterHandler(
  inputName: string,
  registerPasskey: (name: string) => Promise<void>,
) {
  return async () => {
    const trimmedName = inputName.trim()

    if (!trimmedName) {
      Alert.alert('提示', PASSKEY_MESSAGES.registerPrompt)
      return
    }

    await registerPasskey(trimmedName)
  }
}

function PasskeyDemoHeader() {
  return (
    <View style={passkeyStyles.header}>
      <View style={passkeyStyles.iconWrap}>
        <IconSymbol name="faceid" size={22} color="#fff" />
      </View>
      <Text style={passkeyStyles.title}>{PASSKEY_MESSAGES.title}</Text>
    </View>
  )
}

function RegisterActionButton({ status, onRegister }: { status: any; onRegister: () => void }) {
  return (
    <PasskeyActionButton
      label={PASSKEY_MESSAGES.registerButton}
      onPress={onRegister}
      status={status}
      idleIconName="plus.circle.fill"
      successLabel={PASSKEY_MESSAGES.registerButton}
      errorLabel={PASSKEY_MESSAGES.retryButton}
      idleBackgroundColor="#4399fc"
    />
  )
}

function RegisterSection({ inputName, setInputName, status, onRegister }: {
  inputName: string;
  setInputName: (name: string) => void;
  status: any;
  onRegister: () => void;
}) {
  return (
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
      <RegisterActionButton status={status} onRegister={onRegister} />
    </View>
  )
}

function LoginSection({ status, onLogin }: { status: any; onLogin: () => void }) {
  return (
    <View style={passkeyStyles.section}>
      <Text style={passkeyStyles.sectionLabel}>{PASSKEY_MESSAGES.loginSection}</Text>
      <PasskeyActionButton
        label={PASSKEY_MESSAGES.loginButton}
        onPress={onLogin}
        status={status}
        idleIconName="faceid"
        successLabel={PASSKEY_MESSAGES.betaSuccessLabel}
        errorLabel={PASSKEY_MESSAGES.betaRetryLabel}
        idleBackgroundColor="#fa436a"
      />
    </View>
  )
}

function RegisterArea({ userId, inputName, setInputName, status, onRegister }: {
  userId?: number;
  inputName: string;
  setInputName: (name: string) => void;
  status: any;
  onRegister: () => void;
}) {
  if (userId === undefined) return null
  return (
    <>
      <RegisterSection inputName={inputName} setInputName={setInputName} status={status} onRegister={onRegister} />
      <View style={passkeyStyles.divider} />
    </>
  )
}

function PasskeyDemoCard({ userId, inputName, setInputName, status, errorMessage, token, onRegister, onLogin, onReset }: {
  userId?: number;
  inputName: string;
  setInputName: (name: string) => void;
  status: any;
  errorMessage: string | null;
  token: string | null;
  onRegister: () => void;
  onLogin: () => void;
  onReset: () => void;
}) {
  return (
    <View style={passkeyStyles.card}>
      <PasskeyDemoHeader />
      <Text style={passkeyStyles.description}>{PASSKEY_MESSAGES.description}</Text>
      <RegisterArea userId={userId} inputName={inputName} setInputName={setInputName} status={status} onRegister={onRegister} />
      <LoginSection status={status} onLogin={onLogin} />
      <PasskeyStatusNotice
        status={status}
        token={token}
        errorMessage={errorMessage}
        onReset={onReset}
      />
    </View>
  )
}

function useSyncInputName(userName: string | undefined, setInputName: (name: string) => void) {
  useEffect(() => {
    setInputName(userName ?? '')
  }, [userName])
}

function useLoginSuccessNotification(
  onLoginSuccess: Props['onLoginSuccess'],
  status: ReturnType<typeof usePasskey>['status'],
  token: string | null,
) {
  useEffect(() => {
    if (status !== 'success' || !token || !onLoginSuccess) {
      return
    }
    onLoginSuccess(token)
  }, [onLoginSuccess, status, token])
}

function usePasskeyDemoState(userId: number | undefined, userName: string | undefined, onLoginSuccess: Props['onLoginSuccess']) {
  const [inputName, setInputName] = useState(userName ?? '')
  const { status, errorMessage, token, registerPasskey, loginWithPasskey, reset } = usePasskey(userId)
  useSyncInputName(userName, setInputName)
  useLoginSuccessNotification(onLoginSuccess, status, token)
  const handleRegister = createRegisterHandler(inputName, registerPasskey)
  const handleLogin = () => loginWithPasskey()
  return {
    userId, inputName, setInputName, status, errorMessage, token,
    onRegister: handleRegister, onLogin: handleLogin, onReset: reset,
  }
}

export default function PasskeyAuthDemo({ userId, userName, onLoginSuccess }: Props) {
  const passkeyFeatureEnabled = FEATURE_PASSKEY_ENABLED
  const cardProps = usePasskeyDemoState(userId, userName, onLoginSuccess)
  if (!passkeyFeatureEnabled) return null
  return <PasskeyDemoCard {...cardProps} />
}
