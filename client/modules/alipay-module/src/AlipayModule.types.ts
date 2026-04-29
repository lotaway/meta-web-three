export interface AlipayParams {
    orderString: string
}

export interface AlipayEvent {
    type: 'payment_success' | 'payment_error' | 'payment_cancelled'
    message?: string
    data?: any
}

export type AlipayListener = (event: AlipayEvent) => void
