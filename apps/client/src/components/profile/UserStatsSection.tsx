import React from 'react'
import { View, Text, StyleSheet } from 'react-native'
import { useTranslation } from 'react-i18next'
import { Colors } from '@/constants/Colors'
import { useColorScheme } from '@/hooks/useColorScheme'

interface UserStatsSectionProps {
  user: { integration: number; growth: number; couponCount: number }
  colors: any
}

function StatItem({ value, label, colors }: { value: number; label: string; colors: any }) {
  return (
    <View style={styles.statItem}>
      <Text style={[styles.statNum, { color: colors.fontColorDark }]}>{value}</Text>
      <Text style={[styles.statLabel, { color: colors.fontColorLight }]}>{label}</Text>
    </View>
  )
}

export default function UserStatsSection({ user, colors }: UserStatsSectionProps) {
  const { t } = useTranslation()
  const colorScheme = useColorScheme() ?? 'light'
  const themeColors = Colors[colorScheme]
  return (
    <View style={styles.statsSection}>
      <StatItem value={user.integration} label={t('profile.stats.integration')} colors={colors} />
      <StatItem value={user.growth} label={t('profile.stats.growth')} colors={colors} />
      <StatItem value={user.couponCount} label={t('profile.stats.coupon')} colors={colors} />
    </View>
  )
}

const styles = StyleSheet.create({
  statsSection: {
    flexDirection: 'row',
    justifyContent: 'space-around',
    backgroundColor: '#fff',
    paddingVertical: 20,
    marginHorizontal: 15,
    borderRadius: 8,
    marginTop: -20,
    elevation: 4,
    shadowColor: '#000',
    shadowOpacity: 0.1,
    shadowRadius: 10,
    zIndex: 10,
  },
  statItem: {
    alignItems: 'center',
  },
  statNum: {
    fontSize: 18,
    fontWeight: 'bold',
  },
  statLabel: {
    fontSize: 12,
    marginTop: 4,
  },
})
