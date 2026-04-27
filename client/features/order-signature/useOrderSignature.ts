import { useCallback, useState } from 'react'
import Appsdk from 'react-native-appsdk'
import type {
  OrderSignaturePayload,
  OrderSignatureStatus,
  SignedOrderResult,
} from '@/features/order-signature/orderSignature.types'

type OrderSignatureState = {
  status: OrderSignatureStatus
  errorMessage: string | null
  result: SignedOrderResult | null
}

function createIdleState(): OrderSignatureState {
  return { status: 'idle', errorMessage: null, result: null }
}

function createLoadingState(): OrderSignatureState {
  return { status: 'loading', errorMessage: null, result: null }
}

function createSuccessState(result: SignedOrderResult): OrderSignatureState {
  return { status: 'success', errorMessage: null, result }
}

function createErrorState(errorMessage: string): OrderSignatureState {
  return { status: 'error', errorMessage, result: null }
}

export function useOrderSignature(order: OrderSignaturePayload) {
  const [state, setState] = useState<OrderSignatureState>(createIdleState())

  const signOrder = useCallback(async () => {
    setState(createLoadingState())

    try {
      const nonce = await Appsdk.createNonce()
      const timestampMs = await Appsdk.systemTimestampMs()
      const orderData = { ...order, nonce, timestampMs }
      const total = await Appsdk.computeOrderTotal(
        order.price,
        order.quantity,
        order.discount,
        order.freight,
      )
      const signature = await Appsdk.generateRequestSignature(orderData, order.requestKey)

      setState(createSuccessState({ total, signature }))
      return signature
    } catch (error) {
      const errorMessage =
        error instanceof Error ? error.message : '订单签名生成失败'
      setState(createErrorState(errorMessage))
      return null
    }
  }, [order])

  const reset = useCallback(() => {
    setState(createIdleState())
  }, [])

  return {
    ...state,
    signOrder,
    reset,
  }
}
