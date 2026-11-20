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
import ProfileHeader from '@/components/profile/ProfileHeader'
import UserStatsSection from '@/components/profile/UserStatsSection'
import OrderQuickLinksSection from '@/components/profile/OrderQuickLinksSection'
import ProfileMenuCell from '@/components/profile/ProfileMenuCell'

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

type Translate = (key: string) => string

function buildGuestUser(t: Translate): MallUserAccount {
  return {
    nickname: t('profile.nickname_guest'),
    integration: 0,
    growth: 0,
    couponCount: 0,
  }
}

function buildAuthenticatedUser(user: any, t: Translate): MallUserAccount {
  return {
    id: user.id,
    nickname: user.nickname || user.username || t('profile.nickname_guest'),
    avatar: user.avatar,
    integration: (user as any).integration || 0,
    growth: (user as any).growth || 0,
    couponCount: 0,
  } as any
}

async function loadUnreadCount(setUnreadNotifications: (count: number) => void) {
  try {
    const response = await notificationApi.getUnreadCount({ xUserId: DEFAULT_USER_ID })
    if (response.data != null) {
      setUnreadNotifications(response.data as number)
    }
  } catch (error) {
    console.error('Failed to load unread count:', error)
    Alert.alert('Error', 'Failed to load the unread count.')
  }
}

async function loadCouponCount(setCouponCount: (count: number) => void) {
  try {
    const response = await couponApi.listCoupons({})
    if (response.data) {
      setCouponCount((response.data as any[]).length)
    }
  } catch (error) {
    console.error('Failed to load coupon count:', error)
    Alert.alert('Error', 'Failed to load the coupon count.')
  }
}

interface ProfileLoaderSetters {
  setCurrentUser: (user: MallUserAccount) => void
  setIsLoading: (loading: boolean) => void
  setUnreadNotifications: (count: number) => void
  setCouponCount: (count: number) => void
}

async function loadAuthenticatedProfile(userId: number, t: Translate, setters: ProfileLoaderSetters) {
  const response = await userApi.info({ xUserId: userId })
  if (response.data) {
    setters.setCurrentUser(buildAuthenticatedUser(response.data, t))
  }
  loadUnreadCount(setters.setUnreadNotifications)
  loadCouponCount(setters.setCouponCount)
}

async function loadProfileData(
  isAuthenticated: boolean,
  userId: number | null | undefined,
  t: Translate,
  setters: ProfileLoaderSetters,
) {
  setters.setIsLoading(true)
  try {
    if (isAuthenticated && userId) {
      await loadAuthenticatedProfile(userId, t, setters)
    } else {
      setters.setCurrentUser(buildGuestUser(t))
    }
  } catch (error) {
    console.error('Failed to load user profile:', error)
    setters.setCurrentUser(buildGuestUser(t))
  } finally {
    setters.setIsLoading(false)
  }
}

function useProfileData(isAuthenticated: boolean, userId: number | null | undefined, t: Translate) {
  const [isLoading, setIsLoading] = useState(true)
  const [currentUser, setCurrentUser] = useState<MallUserAccount | null>(null)
  const [unreadNotifications, setUnreadNotifications] = useState(0)
  const [couponCount, setCouponCount] = useState(0)

  useEffect(() => {
    const setters = { setCurrentUser, setIsLoading, setUnreadNotifications, setCouponCount }
    loadProfileData(isAuthenticated, userId, t, setters)
  }, [isAuthenticated, userId])

  return { isLoading, currentUser, unreadNotifications, couponCount }
}

function buildLanguageAction(i18n: { changeLanguage: (language: string) => Promise<unknown> }, language: string, label: string) {
  return {
    text: label,
    onPress: async () => {
      await i18n.changeLanguage(language)
      await AsyncStorage.setItem('user-language', language)
    },
  }
}

function buildLanguageActions(i18n: { changeLanguage: (language: string) => Promise<unknown> }, t: Translate) {
  return [
    buildLanguageAction(i18n, 'zh', t('settings.zh')),
    buildLanguageAction(i18n, 'en', t('settings.en')),
    { text: t('common.cancel'), style: 'cancel' as const },
  ]
}

function createPasskeyLoginSuccessHandler(
  userId: number | null | undefined,
  login: (token: string, userId: number) => Promise<void>,
  refreshUser: () => Promise<void>,
) {
  return async (token: string) => {
    if (userId) {
      await login(token, userId)
      await refreshUser()
      Alert.alert('Passkey 登录成功', 'Token 已保存')
    }
  }
}

function createAuthNavigation(authenticated: boolean, router: any, path: string) {
  return () => requireAuth(authenticated, router, () => router.push(path))
}

function PasskeyLoginSection({ userId, userName, onLoginSuccess }: {
  userId: number | undefined;
  userName: string | undefined;
  onLoginSuccess: (token: string) => Promise<void>;
}) {
  if (!FEATURE_PASSKEY_ENABLED) return null
  return (
    <PasskeyAuthDemo
      userId={userId}
      userName={userName}
      onLoginSuccess={onLoginSuccess}
    />
  )
}

interface ProfileMenuSectionProps {
  isAuthenticated: boolean
  router: any
  unreadNotifications: number
  onLanguagePress: () => void
  t: Translate
}

function MainMenuCells({ isAuthenticated, router, unreadNotifications, onLanguagePress, t }: ProfileMenuSectionProps) {
  const go = (path: string) => createAuthNavigation(isAuthenticated, router, path)
  return (
    <>
      <ProfileMenuCell icon="bell.fill" title={t('profile.menu.notification')} color="#FF6B35" onPress={go('/notifications')} badge={unreadNotifications > 0 ? unreadNotifications : undefined} />
      <ProfileMenuCell icon="wallet.pass.fill" title={t('profile.menu.wallet')} color="#FFB300" onPress={go('/wallet')} />
      <ProfileMenuCell icon="ticket.fill" title={t('profile.menu.coupon')} color="#FF3B30" onPress={go('/coupons')} />
      <ProfileMenuCell icon="mappin.and.ellipse" title={t('profile.menu.address')} color="#5fcda2" onPress={go('/address/list')} />
      <ProfileMenuCell icon="clock.fill" title={t('profile.menu.history')} color="#e07472" />
      <ProfileMenuCell icon="star.fill" title={t('profile.menu.following')} color="#5fcda2" />
      <ProfileMenuCell icon="heart.fill" title={t('profile.menu.favorite')} color="#54b4ef" onPress={go('/favorites')} />
      <ProfileMenuCell icon="star.bubble.fill" title={t('profile.menu.comment')} color="#ee883b" />
      <ProfileMenuCell icon="globe" title={t('profile.menu.language')} color="#4a90e2" onPress={onLanguagePress} />
    </>
  )
}

function PasskeyAndSettingsCells({ router, t }: { router: any; t: Translate }) {
  return (
    <>
      {FEATURE_PASSKEY_ENABLED && (
        <ProfileMenuCell
          icon="faceid"
          title="Passkey 安全认证"
          color="#fa436a"
          onPress={() => router.push('/passkey-demo' as any)}
        />
      )}
      <ProfileMenuCell icon="gearshape.fill" title={t('profile.menu.settings')} color="#e07472" showBorder={false} onPress={() => router.push('/settings')} />
    </>
  )
}

function ProfileMenuSection(props: ProfileMenuSectionProps) {
  return (
    <View style={styles.menuSection}>
      <MainMenuCells {...props} />
      <PasskeyAndSettingsCells router={props.router} t={props.t} />
    </View>
  )
}

interface ProfileLoadedContentProps extends ProfileMenuSectionProps {
  currentUser: MallUserAccount | null
  colors: any
  userId: number | null | undefined
  onPasskeyLoginSuccess: (token: string) => Promise<void>
}

type ProfileIdentityProps = Pick<ProfileLoadedContentProps, 'currentUser' | 'colors' | 'isAuthenticated' | 'userId' | 'onPasskeyLoginSuccess'>
type ProfileLinksProps = Pick<ProfileLoadedContentProps, 'colors' | 'isAuthenticated' | 'router' | 'unreadNotifications' | 'onLanguagePress' | 't'>

function ProfileIdentitySections({ currentUser, colors, isAuthenticated, userId, onPasskeyLoginSuccess }: ProfileIdentityProps) {
  return (
    <>
      <ProfileHeader user={currentUser!} colors={colors} />
      <UserStatsSection user={currentUser!} colors={colors} />
      <PasskeyLoginSection
        userId={isAuthenticated ? userId! : undefined}
        userName={currentUser?.nickname}
        onLoginSuccess={onPasskeyLoginSuccess}
      />
    </>
  )
}

function ProfileLinksSections({ colors, isAuthenticated, router, unreadNotifications, onLanguagePress, t }: ProfileLinksProps) {
  return (
    <>
      <OrderQuickLinksSection colors={colors} isAuthenticated={isAuthenticated} router={router} />
      <ProfileMenuSection
        isAuthenticated={isAuthenticated}
        router={router}
        unreadNotifications={unreadNotifications}
        onLanguagePress={onLanguagePress}
        t={t}
      />
    </>
  )
}

function ProfileLoadedContent(props: ProfileLoadedContentProps) {
  return (
    <>
      <ProfileIdentitySections {...props} />
      <ProfileLinksSections {...props} />
    </>
  )
}

function LoadingView({ color }: { color: string }) {
  return (
    <View style={styles.loadingContainer}>
      <ActivityIndicator size="large" color={color} />
    </View>
  )
}

function ProfileScreenView({ colors, isLoading, ...contentProps }: {
  colors: any;
  isLoading: boolean;
} & ProfileLoadedContentProps) {
  return (
    <SafeAreaView style={[styles.container, { backgroundColor: colors.background }]}>
      <ScrollView showsVerticalScrollIndicator={false}>
        {isLoading ? (
          <LoadingView color={colors.primary} />
        ) : (
          <ProfileLoadedContent colors={colors} {...contentProps} />
        )}
        <View style={styles.bottomSpacer} />
      </ScrollView>
    </SafeAreaView>
  )
}

export default function ProfileScreen() {
  const { t, i18n } = useTranslation()
  const colorScheme = useColorScheme() ?? 'light'
  const colors = Colors[colorScheme]
  const router = useRouter()
  const { isAuthenticated, userId, login, refreshUser } = useAuth()
  const { isLoading, currentUser, unreadNotifications, couponCount } = useProfileData(isAuthenticated, userId, t)
  const changeLanguage = () => {
    Alert.alert(t('settings.language_title'), '', buildLanguageActions(i18n, t))
  }
  const handlePasskeyLoginSuccess = createPasskeyLoginSuccessHandler(userId, login, refreshUser)
  const viewProps = { colors, isLoading, currentUser, isAuthenticated, userId, router, unreadNotifications, onLanguagePress: changeLanguage, onPasskeyLoginSuccess: handlePasskeyLoginSuccess, t }
  return <ProfileScreenView {...viewProps} />
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
