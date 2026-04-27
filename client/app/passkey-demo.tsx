import { useState } from 'react'
import {
  Alert,
  SafeAreaView,
  ScrollView,
  StyleSheet,
  Text,
  View,
} from 'react-native'
import PasskeyAuthBeta from '@/components/PasskeyAuthBeta'
import PasskeyAuthDemo from '@/components/PasskeyAuthDemo'
import { IconSymbol } from '@/components/ui/IconSymbol'
import { Colors } from '@/constants/Colors'
import { FEATURE_PASSKEY_ENABLED } from '@/constants/Features'
import { useColorScheme } from '@/hooks/useColorScheme'

const DEMO_USER = {
  id: 1001,
  nickname: 'demo_user',
}

const PASSKEY_FLOW_STEPS = [
  {
    step: '注册',
    icon: 'plus.circle.fill',
    color: '#4399fc',
    description: '后端生成 challenge → 设备生物识别 → 上传 attestation → 后端存储凭证',
  },
  {
    step: '登录',
    icon: 'faceid',
    color: '#fa436a',
    description: '后端生成 challenge → 设备生物识别 → 上传 assertion → 后端验证 → 返回 JWT',
  },
]

function PasskeyDisabledState({
  backgroundColor,
  titleColor,
  descriptionColor,
  iconColor,
}: {
  backgroundColor: string
  titleColor: string
  descriptionColor: string
  iconColor: string
}) {
  return (
    <SafeAreaView style={[styles.safe, { backgroundColor }]}>
      <View style={styles.disabledWrap}>
        <IconSymbol name="lock.slash.fill" size={48} color={iconColor} />
        <Text style={[styles.disabledTitle, { color: titleColor }]}>功能未开放</Text>
        <Text style={[styles.disabledDesc, { color: descriptionColor }]}>
          Passkey 无密码认证功能尚未启用。
          {'\n'}
          如需开启，请在 `.env` 中设置
          {'\n'}
          `EXPO_PUBLIC_PASSKEY_ENABLED=true`
        </Text>
      </View>
    </SafeAreaView>
  )
}

function PasskeyFlowCard({
  cardBackgroundColor,
  titleColor,
  descriptionColor,
}: {
  cardBackgroundColor: string
  titleColor: string
  descriptionColor: string
}) {
  return (
    <View style={[styles.infoCard, { backgroundColor: cardBackgroundColor }]}>
      <Text style={[styles.infoTitle, { color: titleColor }]}>流程说明</Text>
      {PASSKEY_FLOW_STEPS.map(({ step, icon, color, description }) => (
        <View key={step} style={styles.stepRow}>
          <View style={[styles.stepIcon, { backgroundColor: color }]}>
            <IconSymbol name={icon as never} size={14} color="#fff" />
          </View>
          <View style={styles.stepContent}>
            <Text style={[styles.stepTitle, { color: titleColor }]}>{step}</Text>
            <Text style={[styles.stepDesc, { color: descriptionColor }]}>{description}</Text>
          </View>
        </View>
      ))}
    </View>
  )
}

export default function PasskeyDemoScreen() {
  const colorScheme = useColorScheme() ?? 'light'
  const colors = Colors[colorScheme]
  const [lastToken, setLastToken] = useState<string | null>(null)
  const cardBackgroundColor = colors.background === '#f8f8f8' ? '#fff' : '#1e2030'

  if (!FEATURE_PASSKEY_ENABLED) {
    return (
      <PasskeyDisabledState
        backgroundColor={colors.background}
        titleColor={colors.fontColorDark}
        descriptionColor={colors.fontColorLight}
        iconColor={colors.fontColorDisabled}
      />
    )
  }

  return (
    <SafeAreaView style={[styles.safe, { backgroundColor: colors.background }]}>
      <ScrollView contentContainerStyle={styles.scroll} showsVerticalScrollIndicator={false}>
        <View style={styles.pageHeader}>
          <View style={styles.headerIcon}>
            <IconSymbol name="lock.shield.fill" size={28} color="#fff" />
          </View>
          <View style={styles.headerText}>
            <Text style={[styles.pageTitle, { color: colors.fontColorDark }]}>
              Passkey 无密码认证
            </Text>
            <Text style={[styles.pageSubtitle, { color: colors.fontColorLight }]}>
              基于 WebAuthn 的设备生物识别授权示例
            </Text>
          </View>
        </View>

        <PasskeyFlowCard
          cardBackgroundColor={cardBackgroundColor}
          titleColor={colors.fontColorDark}
          descriptionColor={colors.fontColorLight}
        />

        <Text style={[styles.sectionHeader, { color: colors.fontColorBase }]}>完整示例</Text>
        <PasskeyAuthDemo
          userId={DEMO_USER.id}
          userName={DEMO_USER.nickname}
          onLoginSuccess={(token) => {
            setLastToken(token)
            Alert.alert(
              'Passkey 登录成功',
              `JWT Token 已获取\n\n${token.slice(0, 40)}…`,
              [{ text: '好的' }],
            )
          }}
        />

        {lastToken ? (
          <View style={styles.tokenCard}>
            <Text style={styles.tokenLabel}>最近获取的 Token（已截断）</Text>
            <Text style={styles.tokenValue} numberOfLines={2}>
              {lastToken.slice(0, 60)}…
            </Text>
          </View>
        ) : null}

        <Text style={[styles.sectionHeader, { color: colors.fontColorBase }]}>
          场景示例：支付前授权
        </Text>
        <View style={[styles.payCard, { backgroundColor: cardBackgroundColor }]}>
          <View style={styles.payInfo}>
            <Text style={[styles.payTitle, { color: colors.fontColorDark }]}>订单金额</Text>
            <Text style={styles.payAmount}>¥ 299.00</Text>
          </View>
          <Text style={[styles.payHint, { color: colors.fontColorLight }]}>
            请通过生物识别确认支付
          </Text>
          <PasskeyAuthBeta
            label="Face ID 确认支付"
            onAuthorized={() => {
              Alert.alert('支付授权通过', '订单已提交')
            }}
            onError={(message) => {
              Alert.alert('授权失败', message)
            }}
          />
        </View>

        <View style={styles.bottomSpace} />
      </ScrollView>
    </SafeAreaView>
  )
}

const styles = StyleSheet.create({
  safe: { flex: 1 },
  scroll: { paddingBottom: 40 },
  bottomSpace: { height: 50 },
  pageHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 20,
    paddingTop: 24,
    paddingBottom: 8,
    gap: 14,
  },
  headerIcon: {
    width: 52,
    height: 52,
    borderRadius: 16,
    backgroundColor: '#fa436a',
    justifyContent: 'center',
    alignItems: 'center',
  },
  headerText: { flex: 1 },
  pageTitle: { fontSize: 20, fontWeight: '700' },
  pageSubtitle: { fontSize: 13, marginTop: 3 },
  infoCard: {
    marginHorizontal: 15,
    marginTop: 12,
    borderRadius: 14,
    padding: 16,
    shadowColor: '#000',
    shadowOpacity: 0.05,
    shadowRadius: 8,
    elevation: 2,
  },
  infoTitle: {
    fontSize: 14,
    fontWeight: '700',
    marginBottom: 12,
  },
  stepRow: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    marginBottom: 12,
    gap: 10,
  },
  stepIcon: {
    width: 26,
    height: 26,
    borderRadius: 8,
    justifyContent: 'center',
    alignItems: 'center',
    marginTop: 2,
  },
  stepContent: { flex: 1 },
  stepTitle: { fontSize: 13, fontWeight: '600', marginBottom: 2 },
  stepDesc: { fontSize: 12, lineHeight: 17 },
  sectionHeader: {
    fontSize: 12,
    fontWeight: '600',
    textTransform: 'uppercase',
    letterSpacing: 0.6,
    marginHorizontal: 20,
    marginTop: 24,
  },
  tokenCard: {
    marginHorizontal: 15,
    marginTop: 12,
    backgroundColor: '#1a1a2e',
    borderRadius: 10,
    padding: 12,
  },
  tokenLabel: {
    fontSize: 11,
    color: '#909399',
    marginBottom: 6,
  },
  tokenValue: {
    fontSize: 12,
    color: '#4cd964',
    fontFamily: 'monospace',
    lineHeight: 18,
  },
  payCard: {
    marginHorizontal: 15,
    marginTop: 12,
    borderRadius: 14,
    padding: 20,
    shadowColor: '#000',
    shadowOpacity: 0.05,
    shadowRadius: 8,
    elevation: 2,
  },
  payInfo: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 6,
  },
  payTitle: { fontSize: 15, fontWeight: '600' },
  payAmount: { fontSize: 22, fontWeight: '800', color: '#fa436a' },
  payHint: { fontSize: 12, marginBottom: 16 },
  disabledWrap: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    paddingHorizontal: 40,
    gap: 16,
  },
  disabledTitle: {
    fontSize: 18,
    fontWeight: '700',
    marginTop: 8,
  },
  disabledDesc: {
    fontSize: 13,
    lineHeight: 20,
    textAlign: 'center',
  },
})
