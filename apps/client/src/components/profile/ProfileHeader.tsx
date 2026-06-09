import React from 'react'
import { View, Text, StyleSheet, TouchableOpacity } from 'react-native'
import { IconSymbol } from '@/components/ui/IconSymbol'
import { useTranslation } from 'react-i18next'
import { Colors } from '@/constants/Colors'
import { useColorScheme } from '@/hooks/useColorScheme'

interface ProfileHeaderProps {
  user: { nickname: string; avatar?: string }
  colors: any
}

export default function ProfileHeader({ user, colors }: ProfileHeaderProps) {
  const { t } = useTranslation()
  const colorScheme = useColorScheme() ?? 'light'
  const themeColors = Colors[colorScheme]
  return (
    <View style={styles.userSection}>
      <View style={styles.bgGradient} />
      <View style={styles.userInfoBox}>
        <View style={styles.portraitBox}>
          <IconSymbol name="person.circle.fill" size={60} color="#fff" />
        </View>
        <Text style={styles.username}>{user.nickname}</Text>
      </View>

      <View style={styles.vipCardBox}>
        <View style={styles.vipHeader}>
          <View style={styles.vipTitleLine}>
            <IconSymbol name="crown.fill" size={16} color="#f7d680" />
            <Text style={styles.vipTitle}>{t('profile.vip_gold')}</Text>
          </View>
          <TouchableOpacity style={styles.vipBtn}>
            <Text style={styles.vipBtnText}>{t('profile.vip_open_now')}</Text>
          </TouchableOpacity>
        </View>
        <Text style={styles.vipDesc}>{t('profile.vip_slogan')}</Text>
        <Text style={styles.vipFootnote}>{t('profile.vip_note')}</Text>
      </View>
    </View>
  )
}

const styles = StyleSheet.create({
  userSection: {
    height: 260,
    paddingTop: 50,
    paddingHorizontal: 20,
    position: 'relative',
  },
  bgGradient: {
    position: 'absolute',
    left: 0,
    top: 0,
    right: 0,
    bottom: 0,
    backgroundColor: '#306996',
  },
  userInfoBox: {
    flexDirection: 'row',
    alignItems: 'center',
    zIndex: 1,
  },
  portraitBox: {
    width: 60,
    height: 60,
    borderRadius: 30,
    borderWidth: 2,
    borderColor: '#fff',
    overflow: 'hidden',
    justifyContent: 'center',
    alignItems: 'center',
  },
  username: {
    marginLeft: 15,
    fontSize: 20,
    fontWeight: 'bold',
    color: '#fff',
  },
  vipCardBox: {
    marginTop: 30,
    height: 120,
    backgroundColor: 'rgba(0,0,0,0.8)',
    borderRadius: 12,
    padding: 15,
    position: 'relative',
    overflow: 'hidden',
  },
  vipHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  vipTitleLine: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  vipTitle: {
    color: '#f7d680',
    fontSize: 16,
    fontWeight: '500',
    marginLeft: 8,
  },
  vipBtn: {
    backgroundColor: '#f7d680',
    paddingHorizontal: 12,
    paddingVertical: 4,
    borderRadius: 12,
  },
  vipBtnText: {
    color: '#303133',
    fontSize: 12,
    fontWeight: 'bold',
  },
  vipDesc: {
    color: '#f7d680',
    fontSize: 14,
    marginTop: 10,
  },
  vipFootnote: {
    color: '#d8cba9',
    fontSize: 11,
    marginTop: 4,
  },
})
