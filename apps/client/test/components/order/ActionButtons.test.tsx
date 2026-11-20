import { fireEvent, render, screen } from '@testing-library/react-native'
import { ActionButtons } from '@/components/order/ActionButtons'

jest.mock('react-i18next', () => ({
  useTranslation: () => ({ t: (key: string) => `t:${key}` }),
}))

jest.mock('@/hooks/useColorScheme', () => ({
  useColorScheme: () => 'light',
}))

function makeProps() {
  return {
    status: 0,
    orderId: 99,
    onPay: jest.fn(),
    onCancel: jest.fn(),
    onConfirmReceive: jest.fn(),
    onRefund: jest.fn(),
    onContactDelivery: jest.fn(),
    onReview: jest.fn(),
    onContactService: jest.fn(),
  }
}

describe('order/ActionButtons', () => {
  it('always offers customer service', () => {
    const props = makeProps()
    render(<ActionButtons {...props} />)
    expect(screen.getByText('客服')).toBeTruthy()
  })

  it('status 0 shows cancel and pay, wired to their callbacks', () => {
    const props = makeProps()
    render(<ActionButtons {...props} />)

    fireEvent.press(screen.getByText('t:orders.cancel'))
    fireEvent.press(screen.getByText('t:orders.pay'))
    expect(props.onCancel).toHaveBeenCalledTimes(1)
    expect(props.onPay).toHaveBeenCalledTimes(1)
    expect(props.onConfirmReceive).not.toHaveBeenCalled()
  })

  it('status 1 shows refund and contact-delivery, not pay', () => {
    const props = makeProps()
    props.status = 1
    render(<ActionButtons {...props} />)

    fireEvent.press(screen.getByText('t:orders.refund'))
    fireEvent.press(screen.getByText('t:orders.contact_delivery'))
    expect(props.onRefund).toHaveBeenCalledTimes(1)
    expect(props.onContactDelivery).toHaveBeenCalledTimes(1)
    expect(screen.queryByText('t:orders.pay')).toBeNull()
  })

  it('status 2 shows refund and confirm-receive', () => {
    const props = makeProps()
    props.status = 2
    render(<ActionButtons {...props} />)

    fireEvent.press(screen.getByText('t:orders.confirm_receive'))
    expect(props.onConfirmReceive).toHaveBeenCalledTimes(1)
    expect(screen.queryByText('t:orders.pay')).toBeNull()
  })

  it('status 3 shows review and no payment actions', () => {
    const props = makeProps()
    props.status = 3
    render(<ActionButtons {...props} />)

    expect(screen.getByText('评价商品')).toBeTruthy()
    fireEvent.press(screen.getByText('评价商品'))
    expect(props.onReview).toHaveBeenCalledTimes(1)
    expect(props.onPay).not.toHaveBeenCalled()
  })

  it('an unknown status renders only customer service', () => {
    const props = makeProps()
    props.status = 99
    render(<ActionButtons {...props} />)

    expect(screen.getByText('客服')).toBeTruthy()
    expect(screen.queryByText('t:orders.pay')).toBeNull()
    expect(screen.queryByText('t:orders.refund')).toBeNull()
  })
})
