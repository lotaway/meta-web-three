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

export default function LoginScreen() {
  const { t } = useTranslation()
  const router = useRouter()
  const colorScheme = useColorScheme() ?? 'light'
  const colors = Colors[colorScheme]
  const { loginWithCredentials, loginWithSso, loginWithPhone, getAuthCode } = useAuth()

  const [loginType, setLoginType] = useState<'phone' | 'password'>('phone')
  const [phone, setPhone] = useState('')
  const [verificationCode, setVerificationCode] = useState('')
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [loading, setLoading] = useState(false)
  const [countdown, setCountdown] = useState(0)

  const handleSendCode = async () => {
    if (!phone || phone.length < 11) {
      Alert.alert(t('auth.error'), t('auth.phone_invalid'))
      return
    }

    setLoading(true)
    try {
      await getAuthCode(phone)
      
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
      
      Alert.alert(t('auth.success'), t('auth.code_sent'))
    } catch (error) {
      Alert.alert(t('auth.error'), t('auth.code_send_failed'))
    } finally {
      setLoading(false)
    }
  }

  const handleLogin = async () => {
    if (loginType === 'phone') {
      if (!phone || phone.length < 11) {
        Alert.alert(t('auth.error'), t('auth.phone_invalid'))
        return
      }
      if (!verificationCode || verificationCode.length < 4) {
        Alert.alert(t('auth.error'), t('auth.code_invalid'))
        return
      }

      setLoading(true)
      try {
        await loginWithPhone(phone, verificationCode)
        Alert.alert(t('auth.success'), t('auth.login_success'), [
          { text: 'OK', onPress: () => router.replace('/(tabs)') }
        ])
      } catch (error: any) {
        const message = error?.message || t('auth.login_failed')
        Alert.alert(t('auth.error'), message)
      } finally {
        setLoading(false)
      }
    } else {
      if (!username) {
        Alert.alert(t('auth.error'), t('auth.username_required'))
        return
      }
      if (!password || password.length < 6) {
        Alert.alert(t('auth.error'), t('auth.password_invalid'))
        return
      }

      setLoading(true)
      try {
        await loginWithSso(username, password)
        Alert.alert(t('auth.success'), t('auth.login_success'), [
          { text: 'OK', onPress: () => router.replace('/(tabs)') }
        ])
      } catch (error: any) {
        const message = error?.message || t('auth.login_failed')
        Alert.alert(t('auth.error'), message)
      } finally {
        setLoading(false)
      }
    }
  }

  return (
    <SafeAreaView style={[styles.container, { backgroundColor: colors.background }]}>
      <KeyboardAvoidingView
        behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
        style={styles.keyboardView}
      >
        <ScrollView contentContainerStyle={styles.scrollContent}>
          <View style={styles.header}>
            <TouchableOpacity
              style={styles.backBtn}
              onPress={() => router.back()}
            >
              <IconSymbol name="chevron.left" size={24} color={colors.text} />
            </TouchableOpacity>
            <Text style={[styles.title, { color: colors.text }]}>
              {t('auth.login_title')}
            </Text>
          </View>

          <View style={styles.form}>
            <View style={styles.tabContainer}>
              <TouchableOpacity
                style={[
                  styles.tab,
                  loginType === 'phone' && { borderBottomColor: colors.primary }
                ]}
                onPress={() => setLoginType('phone')}
              >
                <Text style={[
                  styles.tabText,
                  { color: loginType === 'phone' ? colors.primary : colors.textSecondary }
                ]}>
                  {t('auth.login_phone')}
                </Text>
              </TouchableOpacity>
              <TouchableOpacity
                style={[
                  styles.tab,
                  loginType === 'password' && { borderBottomColor: colors.primary }
                ]}
                onPress={() => setLoginType('password')}
              >
                <Text style={[
                  styles.tabText,
                  { color: loginType === 'password' ? colors.primary : colors.textSecondary }
                ]}>
                  {t('auth.login_password')}
                </Text>
              </TouchableOpacity>
            </View>

            {loginType === 'phone' ? (
              <View style={styles.fields}>
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

                <View style={styles.field}>
                  <Text style={[styles.label, { color: colors.textSecondary }]}>
                    {t('auth.verification_code')}
                  </Text>
                  <View style={styles.codeRow}>
                    <TextInput
                      style={[styles.codeInput, { color: colors.text, borderColor: colors.border }]}
                      placeholder={t('auth.code_placeholder')}
                      placeholderTextColor={colors.textSecondary}
                      value={verificationCode}
                      onChangeText={setVerificationCode}
                      keyboardType="number-pad"
                      maxLength={6}
                    />
                    <TouchableOpacity
                      style={[
                        styles.codeBtn,
                        { backgroundColor: countdown > 0 ? colors.border : colors.primary }
                      ]}
                      onPress={handleSendCode}
                      disabled={countdown > 0 || loading}
                    >
                      <Text style={styles.codeBtnText}>
                        {countdown > 0 ? `${countdown}s` : t('auth.code_send')}
                      </Text>
                    </TouchableOpacity>
                  </View>
                </View>
              </View>
            ) : (
              <View style={styles.fields}>
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

                <TouchableOpacity 
                  style={styles.forgotLink}
                  onPress={() => router.push('/auth/forgot-password')}
                >
                  <Text style={[styles.forgotText, { color: colors.primary }]}>
                    {t('auth.forgot_password')}
                  </Text>
                </TouchableOpacity>
              </View>
            )}

            <TouchableOpacity
              style={[styles.loginBtn, { backgroundColor: colors.primary }]}
              onPress={handleLogin}
              disabled={loading}
            >
              {loading ? (
                <ActivityIndicator color="#fff" />
              ) : (
                <Text style={styles.loginBtnText}>{t('auth.login_btn')}</Text>
              )}
            </TouchableOpacity>

            <View style={styles.divider}>
              <View style={[styles.line, { backgroundColor: colors.border }]} />
              <Text style={[styles.dividerText, { color: colors.textSecondary }]}>
                {t('auth.or_login_with')}
              </Text>
              <View style={[styles.line, { backgroundColor: colors.border }]} />
            </View>

            <View style={styles.socialRow}>
              <TouchableOpacity style={[styles.socialBtn, { borderColor: colors.border }]}>
                <IconSymbol name="logo.wechat" size={24} color="#07C160" />
              </TouchableOpacity>
              <TouchableOpacity style={[styles.socialBtn, { borderColor: colors.border }]}>
                <IconSymbol name="logo.apple" size={24} color="#000" />
              </TouchableOpacity>
            </View>

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
          </View>
        </ScrollView>
      </KeyboardAvoidingView>
    </SafeAreaView>
  )
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
