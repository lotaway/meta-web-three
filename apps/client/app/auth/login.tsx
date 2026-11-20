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
import { useRouter, Link } from 'expo-router'
import { SafeAreaView } from 'react-native-safe-area-context'
import { useTranslation } from 'react-i18next'
import { Colors } from '@/constants/Colors'
import { useColorScheme } from '@/hooks/useColorScheme'
import { useAuth } from '@/contexts/AuthContext'
import { IconSymbol } from '@/components/ui/IconSymbol'

type Translate = ReturnType<typeof useTranslation>['t']
type AppRouter = ReturnType<typeof useRouter>
type AuthContextValue = ReturnType<typeof useAuth>
type LoginType = 'phone' | 'password'
type SetBoolean = React.Dispatch<React.SetStateAction<boolean>>
type SetCountdown = React.Dispatch<React.SetStateAction<number>>

interface LoginCredentials {
  loginType: LoginType
  phone: string
  verificationCode: string
  username: string
  password: string
}

interface LoginViewProps {
  t: Translate
  colors: typeof Colors.light
  loginType: LoginType
  setLoginType: React.Dispatch<React.SetStateAction<LoginType>>
  phone: string
  setPhone: React.Dispatch<React.SetStateAction<string>>
  verificationCode: string
  setVerificationCode: React.Dispatch<React.SetStateAction<string>>
  username: string
  setUsername: React.Dispatch<React.SetStateAction<string>>
  password: string
  setPassword: React.Dispatch<React.SetStateAction<string>>
  countdown: number
  loading: boolean
  onSendCode: () => void
  onLogin: () => void
  onBack: () => void
  onForgotPassword: () => void
}

interface LoginState {
  loginType: LoginType
  setLoginType: React.Dispatch<React.SetStateAction<LoginType>>
  phone: string
  setPhone: React.Dispatch<React.SetStateAction<string>>
  verificationCode: string
  setVerificationCode: React.Dispatch<React.SetStateAction<string>>
  username: string
  setUsername: React.Dispatch<React.SetStateAction<string>>
  password: string
  setPassword: React.Dispatch<React.SetStateAction<string>>
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

function getPhoneLoginError(credentials: LoginCredentials, t: Translate): string | null {
  if (!isPhoneValid(credentials.phone)) {
    return t('auth.phone_invalid')
  }
  if (!credentials.verificationCode || credentials.verificationCode.length < 4) {
    return t('auth.code_invalid')
  }
  return null
}

function getPasswordLoginError(credentials: LoginCredentials, t: Translate): string | null {
  if (!credentials.username) {
    return t('auth.username_required')
  }
  if (!credentials.password || credentials.password.length < 6) {
    return t('auth.password_invalid')
  }
  return null
}

function getLoginValidationError(credentials: LoginCredentials, t: Translate): string | null {
  return credentials.loginType === 'phone'
    ? getPhoneLoginError(credentials, t)
    : getPasswordLoginError(credentials, t)
}

function showLoginSuccess(t: Translate, router: AppRouter) {
  Alert.alert(t('auth.success'), t('auth.login_success'), [
    { text: 'OK', onPress: () => router.replace('/(tabs)') }
  ])
}

async function runLogin(credentials: LoginCredentials, t: Translate, router: AppRouter, auth: AuthContextValue, setLoading: SetBoolean): Promise<void> {
  setLoading(true)
  try {
    if (credentials.loginType === 'phone') {
      await auth.loginWithPhone(credentials.phone, credentials.verificationCode)
    } else {
      await auth.loginWithSso(credentials.username, credentials.password)
    }
    showLoginSuccess(t, router)
  } catch (error: any) {
    Alert.alert(t('auth.error'), error?.message || t('auth.login_failed'))
  } finally {
    setLoading(false)
  }
}

async function requestAuthCode(
  phone: string,
  t: Translate,
  auth: AuthContextValue,
  setCountdown: SetCountdown
): Promise<void> {
  await auth.getAuthCode(phone)
  startCodeCountdown(setCountdown)
  Alert.alert(t('auth.success'), t('auth.code_sent'))
}

async function sendLoginCode(
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
    await requestAuthCode(phone, t, auth, setCountdown)
  } catch (error) {
    Alert.alert(t('auth.error'), t('auth.code_send_failed'))
  } finally {
    setLoading(false)
  }
}

function useLoginAuth() {
  const { t } = useTranslation()
  const router = useRouter()
  const colors = Colors[useColorScheme() ?? 'light']
  const auth = useAuth()
  return { t, router, colors, auth }
}

function useLoginState(): LoginState {
  const [loginType, setLoginType] = useState<LoginType>('phone')
  const [phone, setPhone] = useState('')
  const [verificationCode, setVerificationCode] = useState('')
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [loading, setLoading] = useState(false)
  const [countdown, setCountdown] = useState(0)
  return {
    loginType, setLoginType, phone, setPhone, verificationCode, setVerificationCode,
    username, setUsername, password, setPassword, loading, setLoading, countdown, setCountdown,
  }
}

function submitLogin(
  credentials: LoginCredentials,
  t: Translate,
  router: AppRouter,
  auth: AuthContextValue,
  setLoading: SetBoolean
) {
  const error = getLoginValidationError(credentials, t)
  if (error) {
    Alert.alert(t('auth.error'), error)
    return
  }
  return runLogin(credentials, t, router, auth, setLoading)
}

function useLoginActions(
  auth: AuthContextValue,
  state: LoginState,
  t: Translate,
  router: AppRouter
) {
  const { phone, verificationCode, username, password, loginType, setLoading, setCountdown } = state
  const credentials = { loginType, phone, verificationCode, username, password }
  const onSendCode = () => sendLoginCode(phone, t, auth, setLoading, setCountdown)
  const onLogin = () => submitLogin(credentials, t, router, auth, setLoading)
  const onBack = () => router.back()
  const onForgotPassword = () => router.push('/auth/forgot-password')
  return { onSendCode, onLogin, onBack, onForgotPassword }
}

function AuthHeader({ colors, t, onBack }: LoginViewProps) {
  return (
    <View style={styles.header}>
      <TouchableOpacity style={styles.backBtn} onPress={onBack}>
        <IconSymbol name="chevron.left" size={24} color={colors.text} />
      </TouchableOpacity>
      <Text style={[styles.title, { color: colors.text }]}>
        {t('auth.login_title')}
      </Text>
    </View>
  )
}

function TabButton({ label, isActive, colors, onPress }: {
  label: string; isActive: boolean; colors: typeof Colors.light; onPress: () => void
}) {
  return (
    <TouchableOpacity
      style={[styles.tab, isActive && { borderBottomColor: colors.primary }]}
      onPress={onPress}
    >
      <Text style={[styles.tabText, { color: isActive ? colors.primary : colors.textSecondary }]}>
        {label}
      </Text>
    </TouchableOpacity>
  )
}

function LoginTabs({ colors, loginType, setLoginType, t }: LoginViewProps) {
  return (
    <View style={styles.tabContainer}>
      <TabButton
        label={t('auth.login_phone')}
        isActive={loginType === 'phone'}
        colors={colors}
        onPress={() => setLoginType('phone')}
      />
      <TabButton
        label={t('auth.login_password')}
        isActive={loginType === 'password'}
        colors={colors}
        onPress={() => setLoginType('password')}
      />
    </View>
  )
}

function PhoneNumberField({ colors, phone, setPhone, t }: LoginViewProps) {
  return (
    <View style={styles.field}>
      <Text style={[styles.label, { color: colors.textSecondary }]}>
        {t('auth.phone_number')}
      </Text>
      <TextInput
        style={[styles.input, { color: colors.text, borderColor: colors.border }]}
        placeholder={t('auth.phone_placeholder')}
        placeholderTextColor={colors.textSecondary}
        value={phone}
        onChangeText={setPhone}
        keyboardType="phone-pad"
        maxLength={11}
      />
    </View>
  )
}

function CodeSendButton({ colors, countdown, loading, onSendCode, t }: LoginViewProps) {
  return (
    <TouchableOpacity
      style={[styles.codeBtn, { backgroundColor: countdown > 0 ? colors.border : colors.primary }]}
      onPress={onSendCode}
      disabled={countdown > 0 || loading}
    >
      <Text style={styles.codeBtnText}>
        {countdown > 0 ? `${countdown}s` : t('auth.code_send')}
      </Text>
    </TouchableOpacity>
  )
}

function CodeInput({ colors, verificationCode, setVerificationCode, t }: LoginViewProps) {
  return (
    <TextInput
      style={[styles.codeInput, { color: colors.text, borderColor: colors.border }]}
      placeholder={t('auth.code_placeholder')}
      placeholderTextColor={colors.textSecondary}
      value={verificationCode}
      onChangeText={setVerificationCode}
      keyboardType="number-pad"
      maxLength={6}
    />
  )
}

function VerificationCodeField(props: LoginViewProps) {
  const { colors, t } = props
  return (
    <View style={styles.field}>
      <Text style={[styles.label, { color: colors.textSecondary }]}>
        {t('auth.verification_code')}
      </Text>
      <View style={styles.codeRow}>
        <CodeInput {...props} />
        <CodeSendButton {...props} />
      </View>
    </View>
  )
}

function PhoneLoginFields(props: LoginViewProps) {
  return (
    <View style={styles.fields}>
      <PhoneNumberField {...props} />
      <VerificationCodeField {...props} />
    </View>
  )
}

function UsernameField({ colors, username, setUsername, t }: LoginViewProps) {
  return (
    <View style={styles.field}>
      <Text style={[styles.label, { color: colors.textSecondary }]}>
        {t('auth.username')}
      </Text>
      <TextInput
        style={[styles.input, { color: colors.text, borderColor: colors.border }]}
        placeholder={t('auth.username_placeholder')}
        placeholderTextColor={colors.textSecondary}
        value={username}
        onChangeText={setUsername}
        autoCapitalize="none"
      />
    </View>
  )
}

function PasswordField({ colors, password, setPassword, t }: LoginViewProps) {
  return (
    <View style={styles.field}>
      <Text style={[styles.label, { color: colors.textSecondary }]}>
        {t('auth.password')}
      </Text>
      <TextInput
        style={[styles.input, { color: colors.text, borderColor: colors.border }]}
        placeholder={t('auth.password_placeholder')}
        placeholderTextColor={colors.textSecondary}
        value={password}
        onChangeText={setPassword}
        secureTextEntry
        autoCapitalize="none"
      />
    </View>
  )
}

function ForgotPasswordLink({ colors, onForgotPassword, t }: LoginViewProps) {
  return (
    <TouchableOpacity style={styles.forgotLink} onPress={onForgotPassword}>
      <Text style={[styles.forgotText, { color: colors.primary }]}>
        {t('auth.forgot_password')}
      </Text>
    </TouchableOpacity>
  )
}

function PasswordLoginFields(props: LoginViewProps) {
  return (
    <View style={styles.fields}>
      <UsernameField {...props} />
      <PasswordField {...props} />
      <ForgotPasswordLink {...props} />
    </View>
  )
}

function LoginSubmitButton({ colors, loading, onLogin, t }: LoginViewProps) {
  return (
    <TouchableOpacity
      style={[styles.loginBtn, { backgroundColor: colors.primary }]}
      onPress={onLogin}
      disabled={loading}
    >
      {loading ? (
        <ActivityIndicator color="#fff" />
      ) : (
        <Text style={styles.loginBtnText}>{t('auth.login_btn')}</Text>
      )}
    </TouchableOpacity>
  )
}

function LoginDivider({ colors, t }: LoginViewProps) {
  return (
    <View style={styles.divider}>
      <View style={[styles.line, { backgroundColor: colors.border }]} />
      <Text style={[styles.dividerText, { color: colors.textSecondary }]}>
        {t('auth.or_login_with')}
      </Text>
      <View style={[styles.line, { backgroundColor: colors.border }]} />
    </View>
  )
}

function SocialLoginButtons({ colors }: LoginViewProps) {
  return (
    <View style={styles.socialRow}>
      <TouchableOpacity style={[styles.socialBtn, { borderColor: colors.border }]}>
        <IconSymbol name={'logo.wechat' as any} size={24} color="#07C160" />
      </TouchableOpacity>
      <TouchableOpacity style={[styles.socialBtn, { borderColor: colors.border }]}>
        <IconSymbol name={'logo.apple' as any} size={24} color="#000" />
      </TouchableOpacity>
    </View>
  )
}

function LoginFooter({ colors, t }: LoginViewProps) {
  return (
    <View style={styles.footer}>
      <Text style={[styles.footerText, { color: colors.textSecondary }]}>
        {t('auth.no_account')}
      </Text>
      <Link href="/auth/register" asChild>
        <TouchableOpacity>
          <Text style={[styles.registerLink, { color: colors.primary }]}>
            {t('auth.register_now')}
          </Text>
        </TouchableOpacity>
      </Link>
    </View>
  )
}

function LoginForm(props: LoginViewProps) {
  const { loginType } = props
  return (
    <View style={styles.form}>
      <LoginTabs {...props} />
      {loginType === 'phone' ? (
        <PhoneLoginFields {...props} />
      ) : (
        <PasswordLoginFields {...props} />
      )}
      <LoginSubmitButton {...props} />
      <LoginDivider {...props} />
      <SocialLoginButtons {...props} />
      <LoginFooter {...props} />
    </View>
  )
}

function LoginView(props: LoginViewProps) {
  return (
    <SafeAreaView style={[styles.container, { backgroundColor: props.colors.background }]}>
      <KeyboardAvoidingView
        behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
        style={styles.keyboardView}
      >
        <ScrollView contentContainerStyle={styles.scrollContent}>
          <AuthHeader {...props} />
          <LoginForm {...props} />
        </ScrollView>
      </KeyboardAvoidingView>
    </SafeAreaView>
  )
}

export default function LoginScreen() {
  const { t, router, colors, auth } = useLoginAuth()
  const state = useLoginState()
  const actions = useLoginActions(auth, state, t, router)
  return <LoginView t={t} colors={colors} {...state} {...actions} />
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
  tabContainer: { flexDirection: 'row', marginBottom: 30 },
  tab: {
    marginRight: 30,
    paddingBottom: 10,
    borderBottomWidth: 2,
    borderBottomColor: 'transparent',
  },
  tabText: { fontSize: 16, fontWeight: '600' },
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
  forgotLink: { alignSelf: 'flex-end', marginTop: 8 },
  forgotText: { fontSize: 14 },
  loginBtn: {
    paddingVertical: 14,
    borderRadius: 8,
    alignItems: 'center',
    marginBottom: 30,
  },
  loginBtnText: { color: '#fff', fontSize: 16, fontWeight: '600' },
  divider: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 30,
  },
  line: { flex: 1, height: 1 },
  dividerText: { marginHorizontal: 16, fontSize: 14 },
  socialRow: {
    flexDirection: 'row',
    justifyContent: 'center',
    gap: 40,
    marginBottom: 40,
  },
  socialBtn: {
    width: 50,
    height: 50,
    borderRadius: 25,
    borderWidth: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  footer: {
    flexDirection: 'row',
    justifyContent: 'center',
    alignItems: 'center',
  },
  footerText: { fontSize: 14 },
  registerLink: { fontSize: 14, fontWeight: '600', marginLeft: 4 },
})
