import React, { useEffect, useState } from 'react'
import { useTranslation } from 'react-i18next'
import AsyncStorage from '@react-native-async-storage/async-storage'
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  Alert,
  ActivityIndicator,
} from 'react-native'
import { SafeAreaView } from 'react-native-safe-area-context'
import { useRouter } from 'expo-router'
import { Colors } from '@/constants/Colors'
import { useColorScheme } from '@/hooks/useColorScheme'
import { useAuth } from '@/contexts/AuthContext'
import PasskeyAuthDemo from '@/components/PasskeyAuthDemo'
import { FEATURE_PASSKEY_ENABLED } from '@/constants/Features'
import { userApi, notificationApi, couponApi, DEFAULT_USER_ID } from '@/api/generated'
import ProfileHeader from '@/app/components/profile/ProfileHeader'
import UserStatsSection from '@/app/components/profile/UserStatsSection'
import OrderQuickLinksSection from '@/app/components/profile/OrderQuickLinksSection'
import ProfileMenuCell from '@/app/components/profile/ProfileMenuCell'

interface MallUserAccount {
  id?: number
  nickname: string
  avatar?: string
  integration: number
  growth: number
  couponCount: number
}

function requireAuth(authenticated: boolean, router: any, callback: () => void) {
  if (!authenticated) {
    router.push('/auth/login')
    return
  }
  callback()
}

export default function ProfileScreen() {
  const { t, i18n } = useTranslation()
  const colorScheme = useColorScheme() ?? 'light'
  const colors = Colors[colorScheme]
  const router = useRouter()
  const { isAuthenticated, userId, login, refreshUser } = useAuth()
  const [isLoading, setIsLoading] = useState(true)
  const [currentUser, setCurrentUser] = useState<MallUserAccount | null>(null)
  const [unreadNotifications, setUnreadNotifications] = useState(0)
  const [couponCount, setCouponCount] = useState(0)

  useEffect(() => {
    loadUserProfile()
  }, [isAuthenticated, userId])

  async function loadUserProfile() {
    setIsLoading(true)
    try {
      if (isAuthenticated && userId) {
        const response = await userApi.info({ xUserId: userId })
        if (response.data) {
          const user = response.data
          setCurrentUser({
            id: user.id,
            nickname: user.nickname || user.username || t('profile.nickname_guest'),
            avatar: user.avatar,
            integration: user.integration || 0,
            growth: 0,
            couponCount: 0,
          })
        }
        loadUnreadCount()
        loadCouponCount()
      } else {
        setCurrentUser({
          nickname: t('profile.nickname_guest'),
          integration: 0,
          growth: 0,
          couponCount: 0,
        })
      }
    } catch (error) {
      console.error('Failed to load user profile:', error)
      setCurrentUser({
        nickname: t('profile.nickname_guest'),
        integration: 0,
        growth: 0,
        couponCount: 0,
      })
    } finally {
      setIsLoading(false)
    }
  }

  async function loadUnreadCount() {
    try {
      const response = await notificationApi.unreadCount({ xUserId: DEFAULT_USER_ID })
      if (response.data != null) {
        setUnreadNotifications(response.data as number)
      }
    } catch (error) {
      console.error('Failed to load unread count:', error)
    }
  }

  async function loadCouponCount() {
    try {
      const response = await couponApi.list({ xUserId: DEFAULT_USER_ID })
      if (response.data) {
        setCouponCount((response.data as any[]).length)
      }
    } catch (error) {
      console.error('Failed to load coupon count:', error)
    }
  }

  const changeLanguage = () => {
    Alert.alert(
      t('settings.language_title'),
      '',
      [
        {
          text: t('settings.zh'),
          onPress: async () => {
            await i18n.changeLanguage('zh')
            await AsyncStorage.setItem('user-language', 'zh')
          },
        },
        {
          text: t('settings.en'),
          onPress: async () => {
            await i18n.changeLanguage('en')
            await AsyncStorage.setItem('user-language', 'en')
          },
        },
        {
          text: t('common.cancel'),
          style: 'cancel',
        },
      ],
    )
  }

  return (
    <SafeAreaView style={[styles.container, { backgroundColor: colors.background }]}>
      <ScrollView showsVerticalScrollIndicator={false}>
        {isLoading ? (
          <View style={styles.loadingContainer}>
            <ActivityIndicator size="large" color={colors.primary} />
          </View>
        ) : (
          <>
            <ProfileHeader user={currentUser!} colors={colors} />

            <UserStatsSection user={currentUser!} colors={colors} />

            {FEATURE_PASSKEY_ENABLED && (
              <PasskeyAuthDemo
                userId={isAuthenticated ? userId! : undefined}
                userName={currentUser?.nickname}
                onLoginSuccess={async (token) => {
                  if (userId) {
                    await login(token, userId)
                    await refreshUser()
                    Alert.alert('Passkey 登录成功', 'Token 已保存')
                  }
                }}
              />
            )}

            <OrderQuickLinksSection colors={colors} isAuthenticated={isAuthenticated} router={router} />

            <View style={styles.menuSection}>
              <ProfileMenuCell icon="bell.fill" title={t('profile.menu.notification')} color="#FF6B35" onPress={() => requireAuth(isAuthenticated, router, () => router.push('/notifications'))} badge={unreadNotifications > 0 ? unreadNotifications : undefined} />
              <ProfileMenuCell icon="wallet.pass.fill" title={t('profile.menu.wallet')} color="#FFB300" onPress={() => requireAuth(isAuthenticated, router, () => router.push('/wallet'))} />
              <ProfileMenuCell icon="ticket.fill" title={t('profile.menu.coupon')} color="#FF3B30" onPress={() => requireAuth(isAuthenticated, router, () => router.push('/coupons'))} />
              <ProfileMenuCell icon="mappin.and.ellipse" title={t('profile.menu.address')} color="#5fcda2" onPress={() => requireAuth(isAuthenticated, router, () => router.push('/address/list'))} />
              <ProfileMenuCell icon="clock.fill" title={t('profile.menu.history')} color="#e07472" />
              <ProfileMenuCell icon="star.fill" title={t('profile.menu.following')} color="#5fcda2" />
              <ProfileMenuCell icon="heart.fill" title={t('profile.menu.favorite')} color="#54b4ef" onPress={() => requireAuth(isAuthenticated, router, () => router.push('/favorites'))} />
              <ProfileMenuCell icon="star.bubble.fill" title={t('profile.menu.comment')} color="#ee883b" />
              <ProfileMenuCell icon="globe" title={t('profile.menu.language')} color="#4a90e2" onPress={changeLanguage} />
              {FEATURE_PASSKEY_ENABLED && (
                <ProfileMenuCell
                  icon="faceid"
                  title="Passkey 安全认证"
                  color="#fa436a"
                  onPress={() => router.push('/passkey-demo' as any)}
                />
              )}
              <ProfileMenuCell icon="gearshape.fill" title={t('profile.menu.settings')} color="#e07472" showBorder={false} onPress={() => router.push('/settings')} />
            </View>
          </>
        )}

        <View style={styles.bottomSpacer} />
      </ScrollView>
    </SafeAreaView>
  )
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  loadingContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    paddingTop: 100,
  },
  menuSection: {
    backgroundColor: '#fff',
    marginHorizontal: 15,
    marginTop: 15,
    borderRadius: 8,
    paddingVertical: 5,
  },
  bottomSpacer: {
    height: 40,
  },
})
