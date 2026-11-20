import React from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  Switch,
  Alert,
} from 'react-native';
import { useRouter } from 'expo-router';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useTranslation } from 'react-i18next';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { Colors } from '@/constants/Colors';
import { useColorScheme } from '@/hooks/useColorScheme';
import { IconSymbol } from '@/components/ui/IconSymbol';
import { useAuth } from '@/contexts/AuthContext';

type AppColors = typeof Colors.light;
type AppRouter = ReturnType<typeof useRouter>;
type Translate = ReturnType<typeof useTranslation>['t'];
type I18nInstance = ReturnType<typeof useTranslation>['i18n'];
type AuthLogout = ReturnType<typeof useAuth>['logout'];

function confirmLogout(logout: AuthLogout, router: AppRouter) {
  Alert.alert('退出登录', '确定要退出登录吗？', [
    { text: '取消', style: 'cancel' },
    { text: '确定', style: 'destructive', onPress: async () => {
      await logout();
      router.back();
    } },
  ]);
}

function applyLanguage(language: string, i18n: I18nInstance) {
  return async () => {
    await i18n.changeLanguage(language);
    await AsyncStorage.setItem('user-language', language);
  };
}

function changeAppLanguage(i18n: I18nInstance, t: Translate) {
  Alert.alert(t('settings.language_title'), '', [
    { text: t('settings.zh'), onPress: applyLanguage('zh', i18n) },
    { text: t('settings.en'), onPress: applyLanguage('en', i18n) },
    { text: t('common.cancel'), style: 'cancel' },
  ]);
}

function showClearCacheHint() {
  Alert.alert('提示', '缓存已清除');
}

function useSettingsController() {
  const { t, i18n } = useTranslation();
  const router = useRouter();
  const colorScheme = useColorScheme() ?? 'light';
  const colors = Colors[colorScheme];
  const { isAuthenticated, logout } = useAuth();
  const [pushEnabled, setPushEnabled] = React.useState(true);
  return {
    t, i18n, router, colors, isAuthenticated, pushEnabled, setPushEnabled,
    handleBack: () => router.back(),
    handleLogout: () => confirmLogout(logout, router),
    changeLanguage: () => changeAppLanguage(i18n, t),
    handleClearCache: showClearCacheHint,
  };
}

type SettingsController = ReturnType<typeof useSettingsController>;

interface SettingsViewProps extends SettingsController {}

function SettingsHeader({ colors, handleBack }: Pick<SettingsViewProps, 'colors' | 'handleBack'>) {
  return (
    <View style={[styles.header, { borderBottomColor: colors.border }]}>
      <TouchableOpacity style={styles.backBtn} onPress={handleBack}>
        <IconSymbol name="chevron.left" size={24} color={colors.text} />
      </TouchableOpacity>
      <Text style={[styles.headerTitle, { color: colors.text }]}>设置</Text>
      <View style={styles.headerRight} />
    </View>
  );
}

function MenuRow({ icon, color, label, colors, onPress, showBorder, right }: {
  icon: string; color: string; label: string; colors: AppColors
  onPress?: () => void; showBorder?: boolean; right?: React.ReactNode
}) {
  const style = showBorder ? [styles.menuItem, { borderBottomColor: colors.border }] : styles.menuItem;
  const content = (
    <>
      <IconSymbol name={icon as any} size={22} color={color} />
      <Text style={[styles.menuText, { color: colors.text }]}>{label}</Text>
      {right ?? <IconSymbol name="chevron.right" size={16} color={colors.textSecondary} />}
    </>
  );
  return onPress
    ? <TouchableOpacity style={style} onPress={onPress}>{content}</TouchableOpacity>
    : <View style={style}>{content}</View>;
}

function ProfileMenuSection({ colors, router }: Pick<SettingsViewProps, 'colors' | 'router'>) {
  return (
    <View style={[styles.section, { backgroundColor: colors.card }]}>
      <MenuRow icon="person.fill" color="#4a90e2" label="个人资料" colors={colors} showBorder onPress={() => router.push('/userinfo')} />
      <MenuRow icon="mappin.and.ellipse" color="#5fcda2" label="收货地址" colors={colors} showBorder onPress={() => router.push('/address/list')} />
      <MenuRow icon="person.badge.shield.checkmark" color="#FF6B35" label="实名认证" colors={colors} />
    </View>
  );
}

function LanguageRight({ i18n, colors }: Pick<SettingsViewProps, 'i18n' | 'colors'>) {
  return (
    <View style={styles.menuRight}>
      <Text style={[styles.menuHint, { color: colors.textSecondary }]}>
        {i18n.language === 'zh' ? '简体中文' : 'English'}
      </Text>
      <IconSymbol name="chevron.right" size={16} color={colors.textSecondary} />
    </View>
  );
}

function AboutRight({ colors }: Pick<SettingsViewProps, 'colors'>) {
  return (
    <View style={styles.menuRight}>
      <Text style={[styles.menuHint, { color: colors.textSecondary }]}>v1.0.0</Text>
      <IconSymbol name="chevron.right" size={16} color={colors.textSecondary} />
    </View>
  );
}

function PushRow({ colors, pushEnabled, setPushEnabled }: Pick<SettingsViewProps, 'colors' | 'pushEnabled' | 'setPushEnabled'>) {
  return (
    <View style={[styles.menuItem, { borderBottomColor: colors.border }]}>
      <IconSymbol name="bell.fill" size={22} color="#fa436a" />
      <Text style={[styles.menuText, { color: colors.text }]}>消息推送</Text>
      <Switch value={pushEnabled} onValueChange={setPushEnabled} trackColor={{ false: '#ddd', true: colors.primary }} thumbColor="#fff" />
    </View>
  );
}

function SystemMenuSection(props: SettingsViewProps) {
  const { colors, i18n, pushEnabled, setPushEnabled, changeLanguage, handleClearCache } = props;
  return (
    <View style={[styles.section, { backgroundColor: colors.card }]}>
      <PushRow colors={colors} pushEnabled={pushEnabled} setPushEnabled={setPushEnabled} />
      <MenuRow icon="globe" color="#4a90e2" label="语言设置" colors={colors} showBorder onPress={changeLanguage} right={<LanguageRight i18n={i18n} colors={colors} />} />
      <MenuRow icon="trash.fill" color="#e07472" label="清除缓存" colors={colors} showBorder onPress={handleClearCache} />
      <MenuRow icon="info.circle.fill" color="#2196F3" label="关于" colors={colors} right={<AboutRight colors={colors} />} />
    </View>
  );
}

function LogoutButton({ colors, isAuthenticated, handleLogout }: Pick<SettingsViewProps, 'colors' | 'isAuthenticated' | 'handleLogout'>) {
  if (!isAuthenticated) return null;
  return (
    <TouchableOpacity style={[styles.logoutBtn, { backgroundColor: colors.card }]} onPress={handleLogout}>
      <Text style={styles.logoutText}>退出登录</Text>
    </TouchableOpacity>
  );
}

function SettingsView(props: SettingsViewProps) {
  const { colors, router, handleBack } = props;
  return (
    <SafeAreaView style={[styles.container, { backgroundColor: colors.background }]}>
      <SettingsHeader colors={colors} handleBack={handleBack} />
      <ScrollView showsVerticalScrollIndicator={false}>
        <ProfileMenuSection colors={colors} router={router} />
        <SystemMenuSection {...props} />
        <LogoutButton {...props} />
        <View style={styles.bottomSpacer} />
      </ScrollView>
    </SafeAreaView>
  );
}

export default function SettingsScreen() {
  const controller = useSettingsController();
  return <SettingsView {...controller} />;
}

const styles = StyleSheet.create({
  container: { flex: 1 },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 12,
    paddingVertical: 12,
    borderBottomWidth: StyleSheet.hairlineWidth,
  },
  backBtn: { padding: 8 },
  headerTitle: {
    flex: 1,
    fontSize: 18,
    fontWeight: '600',
    textAlign: 'center',
    marginRight: 40,
  },
  headerRight: { width: 40 },
  section: {
    marginHorizontal: 16,
    marginTop: 16,
    borderRadius: 12,
    overflow: 'hidden',
  },
  menuItem: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: 16,
    borderBottomWidth: StyleSheet.hairlineWidth,
  },
  menuText: {
    flex: 1,
    fontSize: 15,
    marginLeft: 12,
  },
  menuRight: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  menuHint: {
    fontSize: 13,
    marginRight: 4,
  },
  logoutBtn: {
    marginHorizontal: 16,
    marginTop: 24,
    padding: 16,
    borderRadius: 12,
    alignItems: 'center',
  },
  logoutText: {
    fontSize: 16,
    color: '#FF3B30',
    fontWeight: '500',
  },
  bottomSpacer: {
    height: 40,
  },
});
