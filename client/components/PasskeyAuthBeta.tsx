/**
 * PasskeyAuthBeta — 轻量级单步 Passkey 授权按钮
 *
 * 适用于需要「二次确认」的场景，例如：
 *   - 支付前确认身份
 *   - 查看敏感信息前验证
 *   - 订单提交前授权
 *
 * 用法：
 *   <PasskeyAuthBeta
 *     label="确认支付"
 *     onAuthorized={(token) => submitOrder(token)}
 *   />
 */

import React from 'react';
import {
  TouchableOpacity,
  Text,
  ActivityIndicator,
  StyleSheet,
  View,
} from 'react-native';
import { usePasskey } from '@/hooks/usePasskey';
import { IconSymbol } from '@/components/ui/IconSymbol';
import { FEATURE_PASSKEY_ENABLED } from '@/constants/Features';

interface Props {
  /** 按钮文字，默认："生物识别确认" */
  label?: string;
  /** 授权成功后的回调，参数为服务端返回的 JWT token */
  onAuthorized?: (token: string | null) => void;
  /** 授权失败后的回调 */
  onError?: (msg: string) => void;
}

export default function PasskeyAuthBeta({
  label = '生物识别确认',
  onAuthorized,
  onError,
}: Props) {
  // 功能未启用时不渲染任何内容（按钮直接消失）
  if (!FEATURE_PASSKEY_ENABLED) return null;

  const { status, errorMessage, token, loginWithPasskey, reset } = usePasskey();

  const handlePress = async () => {
    await loginWithPasskey();
  };

  // 成功后触发外部回调（只触发一次）
  React.useEffect(() => {
    if (status === 'success') {
      onAuthorized?.(token);
    }
    if (status === 'error' && errorMessage) {
      onError?.(errorMessage);
    }
    // 触发后自动复位，避免重复回调
    if (status === 'success' || status === 'error') {
      const timer = setTimeout(reset, 2500);
      return () => clearTimeout(timer);
    }
  }, [status]);

  const isLoading = status === 'loading';
  const isSuccess = status === 'success';
  const isError = status === 'error';

  return (
    <View>
      <TouchableOpacity
        style={[
          styles.btn,
          isSuccess && styles.btnSuccess,
          isError && styles.btnError,
          isLoading && styles.btnLoading,
        ]}
        onPress={handlePress}
        disabled={isLoading}
        activeOpacity={0.82}
      >
        {isLoading ? (
          <ActivityIndicator color="#fff" size="small" />
        ) : isSuccess ? (
          <>
            <IconSymbol name="checkmark.shield.fill" size={18} color="#fff" />
            <Text style={styles.btnText}>授权成功</Text>
          </>
        ) : isError ? (
          <>
            <IconSymbol name="exclamationmark.triangle.fill" size={18} color="#fff" />
            <Text style={styles.btnText}>点击重试</Text>
          </>
        ) : (
          <>
            <IconSymbol name="faceid" size={18} color="#fff" />
            <Text style={styles.btnText}>{label}</Text>
          </>
        )}
      </TouchableOpacity>

      {isError && errorMessage && (
        <Text style={styles.errorHint}>{errorMessage}</Text>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  btn: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 8,
    backgroundColor: '#fa436a',
    paddingVertical: 13,
    paddingHorizontal: 24,
    borderRadius: 12,
  },
  btnSuccess: {
    backgroundColor: '#4cd964',
  },
  btnError: {
    backgroundColor: '#dd524d',
  },
  btnLoading: {
    opacity: 0.7,
  },
  btnText: {
    color: '#fff',
    fontSize: 15,
    fontWeight: '700',
  },
  errorHint: {
    marginTop: 6,
    fontSize: 12,
    color: '#dd524d',
    textAlign: 'center',
  },
});
