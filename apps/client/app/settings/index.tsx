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

export default function SettingsScreen() {
  const { t, i18n } = useTranslation();
  const router = useRouter();
  const colorScheme = useColorScheme() ?? 'light';
  const colors = Colors[colorScheme];
  const { isAuthenticated, logout } = useAuth();

  const [pushEnabled, setPushEnabled] = React.useState(true);

  const handleLogout = () => {
    Alert.alert(
      '退出登录',
      '确定要退出登录吗？',
      [
        { text: '取消', style: 'cancel' },
        {
          text: '确定',
          style: 'destructive',
          onPress: async () => {
            await logout();
            router.back();
          },
        },
      ],
    );
  };

  const changeLanguage = () => {
    Alert.alert(
      t('settings.language_title'),
      '',
      [
        {
          text: t('settings.zh'),
          onPress: async () => {
            await i18n.changeLanguage('zh');
            await AsyncStorage.setItem('user-language', 'zh');
          },
        },
        {
          text: t('settings.en'),
          onPress: async () => {
            await i18n.changeLanguage('en');
            await AsyncStorage.setItem('user-language', 'en');
          },
        },
        { text: t('common.cancel'), style: 'cancel' },
      ],
    );
  };

  const handleClearCache = () => {
    Alert.alert('提示', '缓存已清除');
  };

  return (
    <SafeAreaView style={[styles.container, { backgroundColor: colors.background }]}>
      <View style={[styles.header, { borderBottomColor: colors.border }]}>
        <TouchableOpacity style={styles.backBtn} onPress={() => router.back()}>
          <IconSymbol name="chevron.left" size={24} color={colors.text} />
        </TouchableOpacity>
        <Text style={[styles.headerTitle, { color: colors.text }]}>设置</Text>
        <View style={styles.headerRight} />
      </View>

      <ScrollView showsVerticalScrollIndicator={false}>
        <View style={[styles.section, { backgroundColor: colors.card }]}>
          <TouchableOpacity
            style={[styles.menuItem, { borderBottomColor: colors.border }]}
            onPress={() => router.push('/userinfo')}
          >
            <IconSymbol name="person.fill" size={22} color="#4a90e2" />
            <Text style={[styles.menuText, { color: colors.text }]}>个人资料</Text>
            <IconSymbol name="chevron.right" size={16} color={colors.textSecondary} />
          </TouchableOpacity>
          <TouchableOpacity
            style={[styles.menuItem, { borderBottomColor: colors.border }]}
            onPress={() => router.push('/address/list')}
          >
            <IconSymbol name="mappin.and.ellipse" size={22} color="#5fcda2" />
            <Text style={[styles.menuText, { color: colors.text }]}>收货地址</Text>
            <IconSymbol name="chevron.right" size={16} color={colors.textSecondary} />
          </TouchableOpacity>
          <TouchableOpacity style={styles.menuItem}>
            <IconSymbol name="person.badge.shield.checkmark" size={22} color="#FF6B35" />
            <Text style={[styles.menuText, { color: colors.text }]}>实名认证</Text>
            <IconSymbol name="chevron.right" size={16} color={colors.textSecondary} />
          </TouchableOpacity>
        </View>

        <View style={[styles.section, { backgroundColor: colors.card }]}>
          <View style={[styles.menuItem, { borderBottomColor: colors.border }]}>
            <IconSymbol name="bell.fill" size={22} color="#fa436a" />
            <Text style={[styles.menuText, { color: colors.text }]}>消息推送</Text>
            <Switch
              value={pushEnabled}
              onValueChange={setPushEnabled}
              trackColor={{ false: '#ddd', true: colors.primary }}
              thumbColor="#fff"
            />
          </View>
          <TouchableOpacity
            style={[styles.menuItem, { borderBottomColor: colors.border }]}
            onPress={changeLanguage}
          >
            <IconSymbol name="globe" size={22} color="#4a90e2" />
            <Text style={[styles.menuText, { color: colors.text }]}>语言设置</Text>
            <View style={styles.menuRight}>
              <Text style={[styles.menuHint, { color: colors.textSecondary }]}>
                {i18n.language === 'zh' ? '简体中文' : 'English'}
              </Text>
              <IconSymbol name="chevron.right" size={16} color={colors.textSecondary} />
            </View>
          </TouchableOpacity>
          <TouchableOpacity
            style={[styles.menuItem, { borderBottomColor: colors.border }]}
            onPress={handleClearCache}
          >
            <IconSymbol name="trash.fill" size={22} color="#e07472" />
            <Text style={[styles.menuText, { color: colors.text }]}>清除缓存</Text>
            <IconSymbol name="chevron.right" size={16} color={colors.textSecondary} />
          </TouchableOpacity>
          <TouchableOpacity style={styles.menuItem}>
            <IconSymbol name="info.circle.fill" size={22} color="#2196F3" />
            <Text style={[styles.menuText, { color: colors.text }]}>关于</Text>
            <View style={styles.menuRight}>
              <Text style={[styles.menuHint, { color: colors.textSecondary }]}>v1.0.0</Text>
              <IconSymbol name="chevron.right" size={16} color={colors.textSecondary} />
            </View>
          </TouchableOpacity>
        </View>

        {isAuthenticated && (
          <TouchableOpacity
            style={[styles.logoutBtn, { backgroundColor: colors.card }]}
            onPress={handleLogout}
          >
            <Text style={styles.logoutText}>退出登录</Text>
          </TouchableOpacity>
        )}

        <View style={styles.bottomSpacer} />
      </ScrollView>
    </SafeAreaView>
  );
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
