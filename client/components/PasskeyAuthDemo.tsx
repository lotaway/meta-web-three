/**
 * PasskeyAuthDemo — Passkey 注册 & 认证完整示例组件
 *
 * 用法：
 *   <PasskeyAuthDemo userId={currentUser.id} userName={currentUser.nickname} />
 *
 * 功能：
 *  • 注册 Passkey（为当前登录用户绑定设备生物凭证）
 *  • 使用 Passkey 登录（触发设备生物识别 → 获取 JWT）
 *  • 显示流程状态与错误信息
 */

import React, { useState } from 'react';
import {
  View,
  Text,
  TouchableOpacity,
  StyleSheet,
  ActivityIndicator,
  TextInput,
  Alert,
} from 'react-native';
import { usePasskey } from '@/hooks/usePasskey';
import { IconSymbol } from '@/components/ui/IconSymbol';
import { FEATURE_PASSKEY_ENABLED } from '@/constants/Features';

interface Props {
  /** 已登录用户 ID；未传则只展示"登录"入口 */
  userId?: number;
  /** 用于注册时的用户名 */
  userName?: string;
  /** 登录成功的回调，携带 JWT token */
  onLoginSuccess?: (token: string) => void;
}

export default function PasskeyAuthDemo({ userId, userName, onLoginSuccess }: Props) {
  // 功能未启用时不渲染任何内容
  if (!FEATURE_PASSKEY_ENABLED) return null;

  const { status, errorMessage, token, registerPasskey, loginWithPasskey, reset } = usePasskey(userId);
  const [inputName, setInputName] = useState(userName ?? '');

  const handleRegister = async () => {
    const name = inputName.trim();
    if (!name) {
      Alert.alert('提示', '请先输入用户名再注册 Passkey');
      return;
    }
    await registerPasskey(name);
  };

  const handleLogin = async () => {
    await loginWithPasskey();
    if (token && onLoginSuccess) {
      onLoginSuccess(token);
    }
  };

  const isLoading = status === 'loading';

  return (
    <View style={styles.card}>
      {/* 标题行 */}
      <View style={styles.header}>
        <View style={styles.iconWrap}>
          <IconSymbol name="faceid" size={22} color="#fff" />
        </View>
        <Text style={styles.title}>Passkey 安全验证</Text>
      </View>

      <Text style={styles.desc}>
        使用设备生物识别（Face ID / Touch ID）作为无密码凭证，更安全、更便捷。
      </Text>

      {/* 注册区（仅已登录用户可见） */}
      {userId !== undefined && (
        <View style={styles.section}>
          <Text style={styles.sectionLabel}>绑定新 Passkey</Text>
          <TextInput
            style={styles.input}
            value={inputName}
            onChangeText={setInputName}
            placeholder="用于标识该凭证的用户名"
            placeholderTextColor="#b0b3ba"
            editable={!isLoading}
          />
          <TouchableOpacity
            style={[styles.btn, styles.btnRegister, isLoading && styles.btnDisabled]}
            onPress={handleRegister}
            disabled={isLoading}
            activeOpacity={0.8}
          >
            {isLoading ? (
              <ActivityIndicator color="#fff" size="small" />
            ) : (
              <>
                <IconSymbol name="plus.circle.fill" size={16} color="#fff" />
                <Text style={styles.btnText}>注册 Passkey</Text>
              </>
            )}
          </TouchableOpacity>
        </View>
      )}

      <View style={styles.divider} />

      {/* 认证区 */}
      <View style={styles.section}>
        <Text style={styles.sectionLabel}>使用 Passkey 授权</Text>
        <TouchableOpacity
          style={[styles.btn, styles.btnLogin, isLoading && styles.btnDisabled]}
          onPress={handleLogin}
          disabled={isLoading}
          activeOpacity={0.8}
        >
          {isLoading ? (
            <ActivityIndicator color="#fff" size="small" />
          ) : (
            <>
              <IconSymbol name="faceid" size={16} color="#fff" />
              <Text style={styles.btnText}>生物识别登录</Text>
            </>
          )}
        </TouchableOpacity>
      </View>

      {/* 状态反馈 */}
      {status === 'success' && (
        <View style={[styles.statusBox, styles.statusSuccess]}>
          <IconSymbol name="checkmark.circle.fill" size={16} color="#4cd964" />
          <Text style={[styles.statusText, { color: '#4cd964' }]}>
            {token ? `授权成功 · JWT 已获取` : '注册成功！Passkey 已绑定到此设备'}
          </Text>
        </View>
      )}

      {status === 'error' && (
        <View style={[styles.statusBox, styles.statusError]}>
          <IconSymbol name="exclamationmark.triangle.fill" size={16} color="#dd524d" />
          <Text style={[styles.statusText, { color: '#dd524d' }]}>{errorMessage}</Text>
          <TouchableOpacity onPress={reset} style={styles.retryBtn}>
            <Text style={styles.retryText}>重试</Text>
          </TouchableOpacity>
        </View>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  card: {
    backgroundColor: '#fff',
    borderRadius: 14,
    padding: 20,
    marginHorizontal: 15,
    marginTop: 15,
    shadowColor: '#000',
    shadowOpacity: 0.07,
    shadowRadius: 12,
    shadowOffset: { width: 0, height: 4 },
    elevation: 4,
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 8,
  },
  iconWrap: {
    width: 36,
    height: 36,
    borderRadius: 10,
    backgroundColor: '#fa436a',
    justifyContent: 'center',
    alignItems: 'center',
    marginRight: 10,
  },
  title: {
    fontSize: 17,
    fontWeight: '700',
    color: '#1a1a2e',
  },
  desc: {
    fontSize: 13,
    color: '#909399',
    lineHeight: 18,
    marginBottom: 16,
  },
  section: {
    marginBottom: 4,
  },
  sectionLabel: {
    fontSize: 12,
    fontWeight: '600',
    color: '#606266',
    marginBottom: 8,
    textTransform: 'uppercase',
    letterSpacing: 0.5,
  },
  input: {
    borderWidth: 1,
    borderColor: '#E4E7ED',
    borderRadius: 8,
    paddingHorizontal: 12,
    paddingVertical: 10,
    fontSize: 14,
    color: '#303133',
    marginBottom: 10,
    backgroundColor: '#fafafa',
  },
  btn: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: 12,
    borderRadius: 10,
    gap: 6,
  },
  btnRegister: {
    backgroundColor: '#4399fc',
  },
  btnLogin: {
    backgroundColor: '#fa436a',
  },
  btnDisabled: {
    opacity: 0.6,
  },
  btnText: {
    color: '#fff',
    fontSize: 15,
    fontWeight: '600',
  },
  divider: {
    height: 1,
    backgroundColor: '#f0f0f0',
    marginVertical: 16,
  },
  statusBox: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    borderRadius: 8,
    padding: 10,
    marginTop: 14,
    gap: 6,
    flexWrap: 'wrap',
  },
  statusSuccess: {
    backgroundColor: '#f0fff4',
  },
  statusError: {
    backgroundColor: '#fff2f2',
  },
  statusText: {
    fontSize: 13,
    flex: 1,
    lineHeight: 18,
  },
  retryBtn: {
    marginTop: 4,
  },
  retryText: {
    fontSize: 13,
    color: '#fa436a',
    fontWeight: '600',
  },
});
