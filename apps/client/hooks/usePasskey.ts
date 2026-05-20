import { useCallback, useState } from 'react'
import Appsdk from 'react-native-appsdk'
import { passkeyApi, RP_ID } from '@/api/generated'
import { getPasskeyErrorMessage } from '@/features/passkey/passkey.errors'
import { PASSKEY_MESSAGES } from '@/features/passkey/passkey.messages'
import {
  createDisabledPasskeyResult,
  createErrorPasskeyState,
  createIdlePasskeyState,
  createLoadingPasskeyState,
  createSuccessPasskeyState,
} from '@/features/passkey/passkey.state'
import type { PasskeyHookResult, PasskeyState } from '@/features/passkey/passkey.types'
import { FEATURE_PASSKEY_ENABLED } from '@/constants/Features'

function parsePasskeyPayload(payload: string) {
  return JSON.parse(payload)
}

type PasskeyNativeModule = {
  createPasskey: (rpId: string, userName: string) => Promise<string>
  authenticatePasskey: (rpId: string, challenge: string) => Promise<string>
}

const passkeyNativeModule = Appsdk as unknown as PasskeyNativeModule

export function usePasskey(userId?: number): PasskeyHookResult {
  const [state, setState] = useState<PasskeyState>(createIdlePasskeyState())
  const registerDisabledAction = useCallback(async (_userName: string) => {}, [])

  const reset = useCallback(() => {
    setState(createIdlePasskeyState())
  }, [])

  const registerPasskey = useCallback(
    async (userName: string) => {
      if (!userId) {
        setState(createErrorPasskeyState(PASSKEY_MESSAGES.missingUser))
        return
      }

      setState(createLoadingPasskeyState())

      try {
        const optionsResponse = await passkeyApi.generateRegistrationOptions({
          xUserId: userId,
          rpId: RP_ID,
        })

        if (!optionsResponse?.data) {
          setState(createErrorPasskeyState(PASSKEY_MESSAGES.missingRegistrationOptions))
          return
        }

        const attestationJson = await passkeyNativeModule.createPasskey(RP_ID, userName)
        const attestation = parsePasskeyPayload(attestationJson)

        await passkeyApi.verifyRegistration({
          xUserId: userId,
          rpId: RP_ID,
          requestBody: attestation,
        })

        setState(createSuccessPasskeyState())
      } catch (error) {
        setState(
          createErrorPasskeyState(getPasskeyErrorMessage(error, 'register')),
        )
      }
    },
    [userId],
  )

  const loginWithPasskey = useCallback(async () => {
    if (!FEATURE_PASSKEY_ENABLED) {
      return
    }

    setState(createLoadingPasskeyState())

    try {
      const optionsResponse = await passkeyApi.generateAuthenticationOptions({ rpId: RP_ID })
      const challenge = optionsResponse?.data?.challenge

      if (typeof challenge !== 'string' || challenge.length === 0) {
        setState(createErrorPasskeyState(PASSKEY_MESSAGES.missingAuthenticationChallenge))
        return
      }

      const assertionJson = await passkeyNativeModule.authenticatePasskey(RP_ID, challenge)
      const assertion = parsePasskeyPayload(assertionJson)
      const loginResponse = await passkeyApi.verifyAuthentication({
        rpId: RP_ID,
        requestBody: assertion,
      })
      const token = loginResponse?.data?.token ?? null

      setState(createSuccessPasskeyState(token))
    } catch (error) {
      setState(createErrorPasskeyState(getPasskeyErrorMessage(error, 'login')))
    }
  }, [])

  if (!FEATURE_PASSKEY_ENABLED) {
    return createDisabledPasskeyResult(registerDisabledAction)
  }

  return {
    ...state,
    registerPasskey,
    loginWithPasskey,
    reset,
  }
}

export type { PasskeyStatus, PasskeyState } from '@/features/passkey/passkey.types'
