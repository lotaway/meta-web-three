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
type SetBoolean = React.Dispatch<React.SetStateAction<boolean>>
type ValidationRule = () => string | null

interface RegisterCredentials {
  email: string
  password: string
  confirmPassword: string
  referrerId: string
}

interface RegisterViewProps {
  t: Translate
  colors: typeof Colors.light
  email: string
  setEmail: React.Dispatch<React.SetStateAction<string>>
  password: string
  setPassword: React.Dispatch<React.SetStateAction<string>>
  confirmPassword: string
  setConfirmPassword: React.Dispatch<React.SetStateAction<string>>
  referrerId: string
  setReferrerId: React.Dispatch<React.SetStateAction<string>>
  loading: boolean
  onRegister: () => void
  onBack: () => void
}

interface RegisterState {
  email: string
  setEmail: React.Dispatch<React.SetStateAction<string>>
  password: string
  setPassword: React.Dispatch<React.SetStateAction<string>>
  confirmPassword: string
  setConfirmPassword: React.Dispatch<React.SetStateAction<string>>
  referrerId: string
  setReferrerId: React.Dispatch<React.SetStateAction<string>>
  loading: boolean
  setLoading: SetBoolean
}

function validateEmail(email: string): boolean {
  const re = /^[^\s@]+@[^\s@]+\.[^\s@]+$/
  return re.test(email)
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

function getRegistrationValidationError(
  credentials: RegisterCredentials,
  t: Translate
): string | null {
  const rules: ValidationRule[] = [
    () => (!credentials.email || !validateEmail(credentials.email)
      ? t('auth.email_invalid') : null),
    () => (!credentials.password || credentials.password.length < 6
      ? t('auth.password_invalid') : null),
    () => (credentials.password !== credentials.confirmPassword
      ? t('auth.password_not_match') : null),
  ]
  return firstValidationError(rules)
}

async function showRegistrationSuccess(
  credentials: RegisterCredentials,
  t: Translate,
  router: AppRouter,
  auth: AuthContextValue
): Promise<void> {
  Alert.alert(t('auth.success'), t('auth.register_success'), [
    {
      text: t('auth.go_login'),
      onPress: async () => {
        try {
          await auth.loginWithCredentials(credentials.email, credentials.password)
          router.replace('/(tabs)')
        } catch (loginError) {
          router.replace('/auth/login')
        }
      },
    },
  ])
}

async function runRegistrationTransaction(
  credentials: RegisterCredentials,
  t: Translate,
  router: AppRouter,
  auth: AuthContextValue
): Promise<void> {
  const refId = credentials.referrerId ? parseInt(credentials.referrerId, 10) : undefined
  if (credentials.referrerId && isNaN(refId!)) {
    Alert.alert(t('auth.error'), t('auth.referrer_invalid'))
    return
  }
  await auth.register(credentials.email, credentials.password, refId)
  await showRegistrationSuccess(credentials, t, router, auth)
}

async function runRegistration(
  credentials: RegisterCredentials,
  t: Translate,
  router: AppRouter,
  auth: AuthContextValue,
  setLoading: SetBoolean
): Promise<void> {
  setLoading(true)
  try {
    await runRegistrationTransaction(credentials, t, router, auth)
  } catch (error: any) {
    Alert.alert(t('auth.error'), error?.message || t('auth.register_failed'))
  } finally {
    setLoading(false)
  }
}

function useRegisterAuth() {
  const { t } = useTranslation()
  const router = useRouter()
  const colors = Colors[useColorScheme() ?? 'light']
  const auth = useAuth()
  return { t, router, colors, auth }
}

function useRegisterState(): RegisterState {
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [confirmPassword, setConfirmPassword] = useState('')
  const [referrerId, setReferrerId] = useState('')
  const [loading, setLoading] = useState(false)
  return { email, setEmail, password, setPassword, confirmPassword, setConfirmPassword, referrerId, setReferrerId, loading, setLoading }
}

function useRegisterActions(
  auth: AuthContextValue,
  state: RegisterState,
  t: Translate,
  router: AppRouter
) {
  const { email, password, confirmPassword, referrerId, setLoading } = state
  const credentials = { email, password, confirmPassword, referrerId }
  const onRegister = () => {
    const error = getRegistrationValidationError(credentials, t)
    if (error) {
      Alert.alert(t('auth.error'), error)
      return
    }
    return runRegistration(credentials, t, router, auth, setLoading)
  }
  const onBack = () => router.back()
  return { onRegister, onBack }
}

function RegisterHeader({ colors, t, onBack }: RegisterViewProps) {
  return (
    <View style={styles.header}>
      <TouchableOpacity style={styles.backBtn} onPress={onBack}>
        <IconSymbol name="chevron.left" size={24} color={colors.text} />
      </TouchableOpacity>
      <Text style={[styles.title, { color: colors.text }]}>
        {t('auth.register_title')}
      </Text>
    </View>
  )
}

function EmailField({ colors, email, setEmail, t }: RegisterViewProps) {
  return (
    <View style={styles.field}>
      <Text style={[styles.label, { color: colors.textSecondary }]}>
        {t('auth.email')}
      </Text>
      <TextInput
        style={[styles.input, { color: colors.text, borderColor: colors.border }]}
        placeholder={t('auth.email_placeholder')}
        placeholderTextColor={colors.textSecondary}
        value={email}
        onChangeText={setEmail}
        keyboardType="email-address"
        autoCapitalize="none"
      />
    </View>
  )
}

function PasswordHint({ colors, t }: RegisterViewProps) {
  return (
    <Text style={[styles.hint, { color: colors.textSecondary }]}>
      {t('auth.password_hint')}
    </Text>
  )
}

function PasswordField(props: RegisterViewProps) {
  const { colors, password, setPassword, t } = props
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
      <PasswordHint {...props} />
    </View>
  )
}

function ConfirmPasswordField({ colors, confirmPassword, setConfirmPassword, t }: RegisterViewProps) {
  return (
    <View style={styles.field}>
      <Text style={[styles.label, { color: colors.textSecondary }]}>
        {t('auth.confirm_password')}
      </Text>
      <TextInput
        style={[styles.input, { color: colors.text, borderColor: colors.border }]}
        placeholder={t('auth.confirm_password_placeholder')}
        placeholderTextColor={colors.textSecondary}
        value={confirmPassword}
        onChangeText={setConfirmPassword}
        secureTextEntry
        autoCapitalize="none"
      />
    </View>
  )
}

function ReferrerField({ colors, referrerId, setReferrerId, t }: RegisterViewProps) {
  return (
    <View style={styles.field}>
      <Text style={[styles.label, { color: colors.textSecondary }]}>
        {t('auth.referrer_id')} ({t('auth.optional')})
      </Text>
      <TextInput
        style={[styles.input, { color: colors.text, borderColor: colors.border }]}
        placeholder={t('auth.referrer_id_placeholder')}
        placeholderTextColor={colors.textSecondary}
        value={referrerId}
        onChangeText={setReferrerId}
        keyboardType="number-pad"
      />
    </View>
  )
}

function RegisterSubmitButton({ colors, loading, onRegister, t }: RegisterViewProps) {
  return (
    <TouchableOpacity
      style={[styles.registerBtn, { backgroundColor: colors.primary }]}
      onPress={onRegister}
      disabled={loading}
    >
      {loading ? (
        <ActivityIndicator color="#fff" />
      ) : (
        <Text style={styles.registerBtnText}>{t('auth.register_btn')}</Text>
      )}
    </TouchableOpacity>
  )
}

function RegisterFooter({ colors, t }: RegisterViewProps) {
  return (
    <View style={styles.footer}>
      <Text style={[styles.footerText, { color: colors.textSecondary }]}>
        {t('auth.has_account')}
      </Text>
      <Link href="/auth/login" asChild>
        <TouchableOpacity>
          <Text style={[styles.loginLink, { color: colors.primary }]}>
            {t('auth.login_now')}
          </Text>
        </TouchableOpacity>
      </Link>
    </View>
  )
}

function RegisterForm(props: RegisterViewProps) {
  return (
    <View style={styles.form}>
      <View style={styles.fields}>
        <EmailField {...props} />
        <PasswordField {...props} />
        <ConfirmPasswordField {...props} />
        <ReferrerField {...props} />
      </View>
      <RegisterSubmitButton {...props} />
      <RegisterFooter {...props} />
    </View>
  )
}

function RegisterView(props: RegisterViewProps) {
  return (
    <SafeAreaView style={[styles.container, { backgroundColor: props.colors.background }]}>
      <KeyboardAvoidingView
        behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
        style={styles.keyboardView}
      >
        <ScrollView contentContainerStyle={styles.scrollContent}>
          <RegisterHeader {...props} />
          <RegisterForm {...props} />
        </ScrollView>
      </KeyboardAvoidingView>
    </SafeAreaView>
  )
}

export default function RegisterScreen() {
  const { t, router, colors, auth } = useRegisterAuth()
  const state = useRegisterState()
  const actions = useRegisterActions(auth, state, t, router)
  return <RegisterView t={t} colors={colors} {...state} {...actions} />
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
  hint: { fontSize: 12, marginTop: 4 },
  registerBtn: {
    paddingVertical: 14,
    borderRadius: 8,
    alignItems: 'center',
    marginBottom: 30,
  },
  registerBtnText: { color: '#fff', fontSize: 16, fontWeight: '600' },
  footer: {
    flexDirection: 'row',
    justifyContent: 'center',
    alignItems: 'center',
  },
  footerText: { fontSize: 14 },
  loginLink: { fontSize: 14, fontWeight: '600', marginLeft: 4 },
})
