import React, { useState } from 'react'
import {
  View,
  Text,
  StyleSheet,
  TextInput,
  TouchableOpacity,
  Alert,
  ActivityIndicator,
  KeyboardAvoidingView,
  Platform,
  ScrollView,
} from 'react-native'
import { useRouter } from 'expo-router'
import { SafeAreaView } from 'react-native-safe-area-context'
import { useTranslation } from 'react-i18next'
import { Colors } from '@/constants/Colors'
import { useColorScheme } from '@/hooks/useColorScheme'
import { useAuth } from '@/contexts/AuthContext'
import { IconSymbol } from '@/components/ui/IconSymbol'

type Translate = ReturnType<typeof useTranslation>['t']
type AppRouter = ReturnType<typeof useRouter>
type AuthContextValue = ReturnType<typeof useAuth>
type SetBoolean = React.Dispatch<React.SetStateAction<boolean>>
type SetCountdown = React.Dispatch<React.SetStateAction<number>>
type ValidationRule = () => string | null

interface ResetCredentials {
  phone: string
  verificationCode: string
  newPassword: string
  confirmPassword: string
}

interface ResetViewProps {
  t: Translate
  colors: typeof Colors.light
  phone: string
  setPhone: React.Dispatch<React.SetStateAction<string>>
  verificationCode: string
  setVerificationCode: React.Dispatch<React.SetStateAction<string>>
  newPassword: string
  setNewPassword: React.Dispatch<React.SetStateAction<string>>
  confirmPassword: string
  setConfirmPassword: React.Dispatch<React.SetStateAction<string>>
  countdown: number
  loading: boolean
  onSendCode: () => void
  onSubmit: () => void
  onBack: () => void
}

interface ResetState {
  phone: string
  setPhone: React.Dispatch<React.SetStateAction<string>>
  verificationCode: string
  setVerificationCode: React.Dispatch<React.SetStateAction<string>>
  newPassword: string
  setNewPassword: React.Dispatch<React.SetStateAction<string>>
  confirmPassword: string
  setConfirmPassword: React.Dispatch<React.SetStateAction<string>>
  loading: boolean
  setLoading: SetBoolean
  countdown: number
  setCountdown: SetCountdown
}

function isPhoneValid(phone: string): boolean {
  return !!phone && phone.length >= 11
}

function startCodeCountdown(setCountdown: SetCountdown) {
  setCountdown(60)
  const timer = setInterval(() => {
    setCountdown(prev => {
      if (prev <= 1) {
        clearInterval(timer)
        return 0
      }
      return prev - 1
    })
  }, 1000)
}

function firstValidationError(rules: ValidationRule[]): string | null {
  for (const rule of rules) {
    const error = rule()
    if (error) {
      return error
    }
  }
  return null
}

function getSubmitValidationError(credentials: ResetCredentials, t: Translate): string | null {
  const rules: ValidationRule[] = [
    () => (!isPhoneValid(credentials.phone) ? t('auth.phone_invalid') : null),
    () => (!credentials.verificationCode || credentials.verificationCode.length < 4
      ? t('auth.code_invalid') : null),
    () => (!credentials.newPassword || credentials.newPassword.length < 6
      ? t('auth.password_invalid') : null),
    () => (credentials.newPassword !== credentials.confirmPassword
      ? '两次输入的密码不一致' : null),
  ]
  return firstValidationError(rules)
}

function showResetSuccess(t: Translate, router: AppRouter) {
  Alert.alert(t('auth.success'), '密码重置成功，请登录', [
    { text: 'OK', onPress: () => router.replace('/auth/login') }
  ])
}

async function runPasswordReset(credentials: ResetCredentials, t: Translate, router: AppRouter, auth: AuthContextValue, setLoading: SetBoolean): Promise<void> {
  setLoading(true)
  try {
    await auth.forgotPassword(
      credentials.phone,
      credentials.newPassword,
      credentials.verificationCode
    )
    showResetSuccess(t, router)
  } catch (error: any) {
    Alert.alert(t('auth.error'), error?.message || '密码重置失败')
  } finally {
    setLoading(false)
  }
}

async function requestResetCode(
  phone: string,
  t: Translate,
  auth: AuthContextValue,
  setCountdown: SetCountdown
): Promise<void> {
  await auth.getAuthCode(phone)
  startCodeCountdown(setCountdown)
  Alert.alert(t('auth.success'), t('auth.code_sent'))
}

async function sendResetCode(
  phone: string,
  t: Translate,
  auth: AuthContextValue,
  setLoading: SetBoolean,
  setCountdown: SetCountdown
): Promise<void> {
  if (!isPhoneValid(phone)) {
    Alert.alert(t('auth.error'), t('auth.phone_invalid'))
    return
  }
  setLoading(true)
  try {
    await requestResetCode(phone, t, auth, setCountdown)
  } catch (error) {
    Alert.alert(t('auth.error'), t('auth.code_send_failed'))
  } finally {
    setLoading(false)
  }
}

function useResetAuth() {
  const { t } = useTranslation()
  const router = useRouter()
  const colors = Colors[useColorScheme() ?? 'light']
  const auth = useAuth()
  return { t, router, colors, auth }
}

function useResetState(): ResetState {
  const [phone, setPhone] = useState('')
  const [verificationCode, setVerificationCode] = useState('')
  const [newPassword, setNewPassword] = useState('')
  const [confirmPassword, setConfirmPassword] = useState('')
  const [loading, setLoading] = useState(false)
  const [countdown, setCountdown] = useState(0)
  return {
    phone, setPhone, verificationCode, setVerificationCode,
    newPassword, setNewPassword, confirmPassword, setConfirmPassword,
    loading, setLoading, countdown, setCountdown,
  }
}

function useResetActions(
  auth: AuthContextValue,
  state: ResetState,
  t: Translate,
  router: AppRouter
) {
  const { phone, verificationCode, newPassword, confirmPassword, setLoading, setCountdown } = state
  const credentials = { phone, verificationCode, newPassword, confirmPassword }
  const onSendCode = () => sendResetCode(phone, t, auth, setLoading, setCountdown)
  const onSubmit = () => {
    const error = getSubmitValidationError(credentials, t)
    if (error) {
      Alert.alert(t('auth.error'), error)
      return
    }
    return runPasswordReset(credentials, t, router, auth, setLoading)
  }
  const onBack = () => router.back()
  return { onSendCode, onSubmit, onBack }
}

function ResetHeader({ colors, onBack }: ResetViewProps) {
  return (
    <View style={styles.header}>
      <TouchableOpacity style={styles.backBtn} onPress={onBack}>
        <IconSymbol name="chevron.left" size={24} color={colors.text} />
      </TouchableOpacity>
      <Text style={[styles.title, { color: colors.text }]}>
        忘记密码
      </Text>
    </View>
  )
}

function PhoneNumberField({ colors, phone, setPhone }: ResetViewProps) {
  return (
    <View style={styles.field}>
      <Text style={[styles.label, { color: colors.textSecondary }]}>
        手机号
      </Text>
      <TextInput
        style={[styles.input, { color: colors.text, borderColor: colors.border }]}
        placeholder="请输入手机号"
        placeholderTextColor={colors.textSecondary}
        value={phone}
        onChangeText={setPhone}
        keyboardType="phone-pad"
        maxLength={11}
      />
    </View>
  )
}

function CodeSendButton({ colors, countdown, loading, onSendCode }: ResetViewProps) {
  return (
    <TouchableOpacity
      style={[styles.codeBtn, { backgroundColor: countdown > 0 ? colors.border : colors.primary }]}
      onPress={onSendCode}
      disabled={countdown > 0 || loading}
    >
      <Text style={styles.codeBtnText}>
        {countdown > 0 ? `${countdown}s` : '获取验证码'}
      </Text>
    </TouchableOpacity>
  )
}

function CodeInput({ colors, verificationCode, setVerificationCode }: ResetViewProps) {
  return (
    <TextInput
      style={[styles.codeInput, { color: colors.text, borderColor: colors.border }]}
      placeholder="请输入验证码"
      placeholderTextColor={colors.textSecondary}
      value={verificationCode}
      onChangeText={setVerificationCode}
      keyboardType="number-pad"
      maxLength={6}
    />
  )
}

function VerificationCodeField(props: ResetViewProps) {
  const { colors } = props
  return (
    <View style={styles.field}>
      <Text style={[styles.label, { color: colors.textSecondary }]}>
        验证码
      </Text>
      <View style={styles.codeRow}>
        <CodeInput {...props} />
        <CodeSendButton {...props} />
      </View>
    </View>
  )
}

function NewPasswordField({ colors, newPassword, setNewPassword }: ResetViewProps) {
  return (
    <View style={styles.field}>
      <Text style={[styles.label, { color: colors.textSecondary }]}>
        新密码
      </Text>
      <TextInput
        style={[styles.input, { color: colors.text, borderColor: colors.border }]}
        placeholder="8-18位密码"
        placeholderTextColor={colors.textSecondary}
        value={newPassword}
        onChangeText={setNewPassword}
        secureTextEntry
        autoCapitalize="none"
      />
    </View>
  )
}

function ConfirmPasswordField({ colors, confirmPassword, setConfirmPassword }: ResetViewProps) {
  return (
    <View style={styles.field}>
      <Text style={[styles.label, { color: colors.textSecondary }]}>
        确认密码
      </Text>
      <TextInput
        style={[styles.input, { color: colors.text, borderColor: colors.border }]}
        placeholder="再次输入新密码"
        placeholderTextColor={colors.textSecondary}
        value={confirmPassword}
        onChangeText={setConfirmPassword}
        secureTextEntry
        autoCapitalize="none"
      />
    </View>
  )
}

function ResetSubmitButton({ colors, loading, onSubmit }: ResetViewProps) {
  return (
    <TouchableOpacity
      style={[styles.submitBtn, { backgroundColor: colors.primary }]}
      onPress={onSubmit}
      disabled={loading}
    >
      {loading ? (
        <ActivityIndicator color="#fff" />
      ) : (
        <Text style={styles.submitBtnText}>重置密码</Text>
      )}
    </TouchableOpacity>
  )
}

function ResetForm(props: ResetViewProps) {
  return (
    <View style={styles.form}>
      <View style={styles.fields}>
        <PhoneNumberField {...props} />
        <VerificationCodeField {...props} />
        <NewPasswordField {...props} />
        <ConfirmPasswordField {...props} />
      </View>
      <ResetSubmitButton {...props} />
    </View>
  )
}

function ResetView(props: ResetViewProps) {
  return (
    <SafeAreaView style={[styles.container, { backgroundColor: props.colors.background }]}>
      <KeyboardAvoidingView
        behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
        style={styles.keyboardView}
      >
        <ScrollView contentContainerStyle={styles.scrollContent}>
          <ResetHeader {...props} />
          <ResetForm {...props} />
        </ScrollView>
      </KeyboardAvoidingView>
    </SafeAreaView>
  )
}

export default function ForgotPasswordScreen() {
  const { t, router, colors, auth } = useResetAuth()
  const state = useResetState()
  const actions = useResetActions(auth, state, t, router)
  return <ResetView t={t} colors={colors} {...state} {...actions} />
}

const styles = StyleSheet.create({
  container: { flex: 1 },
  keyboardView: { flex: 1 },
  scrollContent: { flexGrow: 1, paddingHorizontal: 24 },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingTop: 20,
    paddingBottom: 30,
  },
  backBtn: { padding: 8 },
  title: { fontSize: 24, fontWeight: 'bold', marginLeft: 16 },
  form: { flex: 1 },
  fields: { marginBottom: 30 },
  field: { marginBottom: 20 },
  label: { fontSize: 14, marginBottom: 8 },
  input: {
    borderWidth: 1,
    borderRadius: 8,
    paddingHorizontal: 12,
    paddingVertical: 12,
    fontSize: 16,
  },
  codeRow: { flexDirection: 'row', gap: 12 },
  codeInput: {
    flex: 1,
    borderWidth: 1,
    borderRadius: 8,
    paddingHorizontal: 12,
    paddingVertical: 12,
    fontSize: 16,
  },
  codeBtn: {
    justifyContent: 'center',
    alignItems: 'center',
    paddingHorizontal: 16,
    borderRadius: 8,
    minWidth: 100,
  },
  codeBtnText: { color: '#fff', fontSize: 14, fontWeight: '500' },
  submitBtn: {
    paddingVertical: 14,
    borderRadius: 8,
    alignItems: 'center',
    marginBottom: 30,
  },
  submitBtnText: { color: '#fff', fontSize: 16, fontWeight: '600' },
})
