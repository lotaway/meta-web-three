import { useEffect, useRef } from 'react'
import { NativeEventEmitter, NativeModules } from 'react-native'
import NativeWechatPay from './NativeWechatPay'
import { WechatPayEvent } from './WechatPayEvent'

const eventEmitter = new NativeEventEmitter(
    NativeModules.WechatPay ?? NativeWechatPay
)

export interface WechatPayFinalConfirmEvent {
    message: string
    timestamp: number
}

export function addWechatPayListener<T = unknown>(
    eventType: WechatPayEvent,
    listener: (event: T) => void
) {
    return eventEmitter.addListener(eventType, listener)
}

export function useWechatPayEvent<T = unknown>(
    eventType: WechatPayEvent,
    listener: (event: T) => void
) {
    const listenerRef = useRef(listener)
    listenerRef.current = listener

    useEffect(() => {
        const subscription = addWechatPayListener<T>(eventType, (event) => {
            listenerRef.current(event)
        })
        return () => subscription.remove()
    }, [eventType])
}

export function emitFinalConfirmEvent(message: string) {
    NativeWechatPay.emitFinalConfirmEvent(message)
}

export { NativeWechatPay, WechatPayEvent }
export type { WechatPayParams } from './NativeWechatPay'
