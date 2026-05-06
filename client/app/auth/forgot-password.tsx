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

export default function ForgotPasswordScreen() {
  const { t } = useTranslation()
  const router = useRouter()
  const colorScheme = useColorScheme() ?? 'light'
  const colors = Colors[colorScheme]
  const { forgotPassword, getAuthCode } = useAuth()

  const [phone, setPhone] = useState('')
  const [verificationCode, setVerificationCode] = useState('')
  const [newPassword, setNewPassword] = useState('')
  const [confirmPassword, setConfirmPassword] = useState('')
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

  const handleSubmit = async () => {
    if (!phone || phone.length < 11) {
      Alert.alert(t('auth.error'), t('auth.phone_invalid'))
      return
    }
    if (!verificationCode || verificationCode.length < 4) {
      Alert.alert(t('auth.error'), t('auth.code_invalid'))
      return
    }
    if (!newPassword || newPassword.length < 6) {
      Alert.alert(t('auth.error'), t('auth.password_invalid'))
      return
    }
    if (newPassword !== confirmPassword) {
      Alert.alert(t('auth.error'), '两次输入的密码不一致')
      return
    }

    setLoading(true)
    try {
      await forgotPassword(phone, newPassword, verificationCode)
      Alert.alert(t('auth.success'), '密码重置成功，请登录', [
        { text: 'OK', onPress: () => router.replace('/auth/login') }
      ])
    } catch (error: any) {
      const message = error?.message || '密码重置失败'
      Alert.alert(t('auth.error'), message)
    } finally {
      setLoading(false)
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
              忘记密码
            </Text>
          </View>

          <View style={styles.form}>
            <View style={styles.fields}>
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

              <View style={styles.field}>
                <Text style={[styles.label, { color: colors.textSecondary }]}>
                  验证码
                </Text>
                <View style={styles.codeRow}>
                  <TextInput
                    style={[styles.codeInput, { color: colors.text, borderColor: colors.border }]}
                    placeholder="请输入验证码"
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
                      {countdown > 0 ? `${countdown}s` : '获取验证码'}
                    </Text>
                  </TouchableOpacity>
                </View>
              </View>

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
            </View>

            <TouchableOpacity
              style={[styles.submitBtn, { backgroundColor: colors.primary }]}
              onPress={handleSubmit}
              disabled={loading}
            >
              {loading ? (
                <ActivityIndicator color="#fff" />
              ) : (
                <Text style={styles.submitBtnText}>重置密码</Text>
              )}
            </TouchableOpacity>
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
