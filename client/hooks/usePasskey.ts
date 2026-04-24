/**
 * usePasskey — Passkey 注册 & 认证流程 Hook
 *
 * 注册流程 (registerPasskey):
 *   1. 向后端请求注册 options（challenge）
 *   2. 调用设备 Passkey 能力（Appsdk.createPasskey）触发系统 UI
 *   3. 将 attestation 提交后端验证存储
 *
 * 认证流程 (loginWithPasskey):
 *   1. 向后端请求认证 options（challenge）
 *   2. 调用设备 Passkey 能力（Appsdk.authenticatePasskey）触发系统 UI
 *   3. 将 assertion 提交后端验证，拿回 JWT
 */

import { useCallback, useState } from 'react'
import Appsdk from 'react-native-appsdk'
import { passkeyApi, RP_ID } from '@/api/generated'
import { FEATURE_PASSKEY_ENABLED } from '@/constants/Features'

export type PasskeyStatus = 'idle' | 'loading' | 'success' | 'error' | 'disabled'

export interface PasskeyState {
  status: PasskeyStatus
  errorMessage: string | null
  /** 认证成功后的 JWT token */
  token: string | null
}

export function usePasskey(userId?: number) {
  // ── 功能开关守卫 ──────────────────────────────────────────────────────────
  // 当 EXPO_PUBLIC_PASSKEY_ENABLED !== 'true' 时，hook 全部方法均为无操作，
  // 调用方无需额外判断开关状态，直接使用 status === 'disabled' 即可。
  const noop = useCallback(async (..._args: any[]) => {}, [])
  const disabledState: PasskeyState & {
    registerPasskey: (...args: any[]) => Promise<void>
    loginWithPasskey: () => Promise<void>
    reset: () => void
  } = {
    status: 'disabled',
    errorMessage: null,
    token: null,
    registerPasskey: noop,
    loginWithPasskey: noop,
    reset: noop,
  }
  // 注意：React Hooks 规则要求 hooks 调用在条件分支之外，
  // 因此在此处调用 useState 后再根据 flag 决定是否提前返回。
  const [state, setState] = useState<PasskeyState>({
    status: 'idle',
    errorMessage: null,
    token: null,
  })

  const setLoading = () => setState({ status: 'loading', errorMessage: null, token: null })
  const setError = (msg: string) => setState({ status: 'error', errorMessage: msg, token: null })
  const setSuccess = (token: string | null = null) =>
    setState({ status: 'success', errorMessage: null, token })

  if (!FEATURE_PASSKEY_ENABLED) {
    return disabledState
  }

  /** ── 注册 Passkey ── */
  const registerPasskey = useCallback(
    async (userName: string) => {
      if (!userId) {
        setError('用户未登录，无法注册 Passkey')
        return
      }
      setLoading()
      try {
        // Step 1: 从后端获取注册 options（含 challenge）
        const optionsResp = await passkeyApi.generateRegistrationOptions({
          xUserId: userId,
          rpId: RP_ID,
        })

        if (!optionsResp?.data) {
          setError('获取注册选项失败')
          return
        }

        // Step 2: 调用设备 Passkey 能力，触发系统生物识别 UI
        // createPasskey 返回 attestation JSON 字符串
        const attestationJson = await Appsdk.createPasskey(RP_ID, userName)
        const attestation = JSON.parse(attestationJson)

        // Step 3: 将 attestation 提交后端验证并存储
        await passkeyApi.verifyRegistration({
          xUserId: userId,
          rpId: RP_ID,
          requestBody: attestation,
        })

        setSuccess()
      } catch (e: any) {
        const msg =
          e?.code === 'BUSY'
            ? '有其他 Passkey 请求进行中，请稍后再试'
            : e?.code === 'UNSUPPORTED'
              ? '设备不支持 Passkey（需要 iOS 16+）'
              : e?.message ?? '注册 Passkey 失败'
        setError(msg)
      }
    },
    [userId],
  )

  /** ── 使用 Passkey 登录/授权 ── */
  const loginWithPasskey = useCallback(async () => {
    setLoading()
    try {
      // Step 1: 从后端获取认证 options（含 challenge）
      const optionsResp = await passkeyApi.generateAuthenticationOptions({ rpId: RP_ID })

      const challenge: string =
        (optionsResp?.data?.challenge as string) ?? ''

      if (!challenge) {
        setError('获取认证 challenge 失败')
        return
      }

      // Step 2: 调用设备 Passkey 能力，触发系统验证 UI
      // authenticatePasskey 返回 assertion JSON 字符串
      const assertionJson = await Appsdk.authenticatePasskey(RP_ID, challenge)
      const assertion = JSON.parse(assertionJson)

      // Step 3: 将 assertion 提交后端，换取 JWT
      const loginResp = await passkeyApi.verifyAuthentication({
        rpId: RP_ID,
        requestBody: assertion,
      })

      const token = (loginResp?.data as any)?.token ?? null
      setSuccess(token)
    } catch (e: any) {
      const msg =
        e?.code === 'BUSY'
          ? '有其他 Passkey 请求进行中，请稍后再试'
          : e?.code === 'UNSUPPORTED'
            ? '设备不支持 Passkey（需要 iOS 16+）'
            : e?.code === 'AUTH_ERROR'
              ? '验证取消或失败，请重试'
              : e?.message ?? '使用 Passkey 登录失败'
      setError(msg)
    }
  }, [])

  const reset = useCallback(() => {
    setState({ status: 'idle', errorMessage: null, token: null })
  }, [])

  return { ...state, registerPasskey, loginWithPasskey, reset }
}
