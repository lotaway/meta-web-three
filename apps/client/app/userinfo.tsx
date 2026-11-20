import React, { useEffect, useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  TextInput,
  Image,
  ActivityIndicator,
  Alert,
} from 'react-native';
import { useRouter } from 'expo-router';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useTranslation } from 'react-i18next';
import { Colors } from '@/constants/Colors';
import { useColorScheme } from '@/hooks/useColorScheme';
import { IconSymbol } from '@/components/ui/IconSymbol';
import { useAuth } from '@/contexts/AuthContext';
import { userApi, API_BASE_URL } from '@/api/generated';

type AppColors = typeof Colors.light;
type AppRouter = ReturnType<typeof useRouter>;
type AuthUserId = ReturnType<typeof useAuth>['userId'];

async function loadUserInfoData(userId: AuthUserId, state: UserInfoState) {
  state.setLoading(true);
  try {
    const response = await userApi.info({ xUserId: userId! });
    const user = response.data as any;
    if (user) {
      state.setNickname(user.nickname || user.username || '');
      state.setEmail(user.email || '');
      state.setAvatar(user.avatar || '');
    }
  } catch (error) {
    console.error('Failed to load user info:', error);
    Alert.alert('Error', 'Failed to load the user info.');
  } finally {
    state.setLoading(false);
  }
}

async function requestSaveNickname(nickname: string) {
  const response = await fetch(`${API_BASE_URL}/user/info`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ nickname: nickname.trim() }),
  });
  return response.ok;
}

async function saveUserInfo(nickname: string, router: AppRouter, state: UserInfoState) {
  if (!nickname.trim()) {
    Alert.alert('提示', '昵称不能为空');
    return;
  }
  state.setSaving(true);
  try {
    if (await requestSaveNickname(nickname)) {
      Alert.alert('成功', '个人信息已更新');
      router.back();
    } else {
      Alert.alert('失败', '更新失败，请重试');
    }
  } catch (error) {
    Alert.alert('错误', '网络请求失败');
  } finally {
    state.setSaving(false);
  }
}

function useUserInfoState() {
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [nickname, setNickname] = useState('');
  const [email, setEmail] = useState('');
  const [avatar, setAvatar] = useState('');
  return { loading, setLoading, saving, setSaving, nickname, setNickname, email, setEmail, avatar, setAvatar };
}

type UserInfoState = ReturnType<typeof useUserInfoState>;

function useUserInfoController() {
  const { t } = useTranslation();
  const router = useRouter();
  const colorScheme = useColorScheme() ?? 'light';
  const colors = Colors[colorScheme];
  const { isAuthenticated, userId } = useAuth();
  const state = useUserInfoState();
  useEffect(() => {
    if (isAuthenticated && userId) loadUserInfoData(userId, state);
  }, [isAuthenticated, userId]);
  return { t, router, colors, ...state, handleBack: () => router.back(), handleSave: () => saveUserInfo(state.nickname, router, state) };
}

interface UserInfoViewProps extends UserInfoState {
  router: AppRouter;
  colors: AppColors;
  handleBack: () => void;
  handleSave: () => void;
}

function UserInfoHeader({ colors, handleBack, handleSave, saving, showSave }: Pick<UserInfoViewProps, 'colors' | 'handleBack' | 'handleSave'> & { saving: boolean; showSave: boolean }) {
  return (
    <View style={showSave ? [styles.header, { borderBottomColor: colors.border }] : styles.header}>
      <TouchableOpacity style={styles.backBtn} onPress={handleBack}>
        <IconSymbol name="chevron.left" size={24} color={colors.text} />
      </TouchableOpacity>
      <Text style={[styles.headerTitle, { color: colors.text }]}>编辑资料</Text>
      {showSave ? (
        <TouchableOpacity onPress={handleSave} disabled={saving}>
          <Text style={[styles.saveBtn, { color: colors.primary }]}>{saving ? '保存中...' : '保存'}</Text>
        </TouchableOpacity>
      ) : (
        <View style={styles.headerRight} />
      )}
    </View>
  );
}

function LoadingView(props: UserInfoViewProps) {
  const { colors } = props;
  return (
    <SafeAreaView style={[styles.container, { backgroundColor: colors.background }]}>
      <UserInfoHeader {...props} saving={false} showSave={false} />
      <View style={styles.center}>
        <ActivityIndicator size="large" color={colors.tint} />
      </View>
    </SafeAreaView>
  );
}

function AvatarSection({ avatar, colors }: Pick<UserInfoViewProps, 'colors'> & { avatar: string }) {
  return (
    <View style={[styles.avatarSection, { backgroundColor: colors.card }]}>
      <Image source={{ uri: avatar || 'https://via.placeholder.com/80' }} style={styles.avatar} />
      <TouchableOpacity style={styles.avatarEdit}>
        <IconSymbol name="camera.fill" size={18} color="#fff" />
      </TouchableOpacity>
    </View>
  );
}

function FormSection({ nickname, setNickname, email, colors }: Pick<UserInfoViewProps, 'colors' | 'nickname' | 'setNickname' | 'email'>) {
  return (
    <View style={[styles.formSection, { backgroundColor: colors.card }]}>
      <View style={[styles.formRow, { borderBottomColor: colors.border }]}>
        <Text style={[styles.formLabel, { color: colors.text }]}>昵称</Text>
        <TextInput style={[styles.formInput, { color: colors.text }]} value={nickname} onChangeText={setNickname} placeholder="请输入昵称" placeholderTextColor={colors.textSecondary} />
      </View>
      <View style={styles.formRow}>
        <Text style={[styles.formLabel, { color: colors.text }]}>邮箱</Text>
        <Text style={[styles.formValue, { color: colors.textSecondary }]}>{email}</Text>
      </View>
    </View>
  );
}

function UserInfoView(props: UserInfoViewProps) {
  const { colors, nickname, setNickname, email, avatar, saving } = props;
  return (
    <SafeAreaView style={[styles.container, { backgroundColor: colors.background }]}>
      <UserInfoHeader {...props} saving={saving} showSave />
      <ScrollView showsVerticalScrollIndicator={false}>
        <AvatarSection avatar={avatar} colors={colors} />
        <FormSection nickname={nickname} setNickname={setNickname} email={email} colors={colors} />
        <View style={styles.bottomSpacer} />
      </ScrollView>
    </SafeAreaView>
  );
}

export default function UserInfoScreen() {
  const controller = useUserInfoController();
  return controller.loading ? <LoadingView {...controller} /> : <UserInfoView {...controller} />;
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
  saveBtn: {
    fontSize: 16,
    fontWeight: '500',
  },
  center: { flex: 1, justifyContent: 'center', alignItems: 'center' },
  avatarSection: {
    alignItems: 'center',
    paddingVertical: 32,
    marginTop: 16,
    marginHorizontal: 16,
    borderRadius: 12,
  },
  avatar: {
    width: 80,
    height: 80,
    borderRadius: 40,
  },
  avatarEdit: {
    position: 'absolute',
    right: '42%',
    bottom: 28,
    backgroundColor: 'rgba(0,0,0,0.5)',
    width: 28,
    height: 28,
    borderRadius: 14,
    justifyContent: 'center',
    alignItems: 'center',
  },
  formSection: {
    marginHorizontal: 16,
    marginTop: 16,
    borderRadius: 12,
  },
  formRow: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: 16,
    borderBottomWidth: StyleSheet.hairlineWidth,
  },
  formLabel: {
    fontSize: 15,
    width: 60,
  },
  formInput: {
    flex: 1,
    fontSize: 15,
    padding: 0,
  },
  formValue: {
    fontSize: 15,
    flex: 1,
  },
  bottomSpacer: {
    height: 40,
  },
});
