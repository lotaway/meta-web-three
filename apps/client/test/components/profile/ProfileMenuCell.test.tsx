import { fireEvent, render, screen } from '@testing-library/react-native'
import ProfileMenuCell from '@/components/profile/ProfileMenuCell'

jest.mock('@/components/ui/IconSymbol', () => ({
  IconSymbol: 'IconSymbol',
}))

jest.mock('@/hooks/useColorScheme', () => ({
  useColorScheme: () => 'light',
}))

describe('profile/ProfileMenuCell', () => {
  it('renders the title and calls onPress when tapped', () => {
    const onPress = jest.fn()
    render(
      <ProfileMenuCell
        icon="person"
        title="Personal Info"
        color="#000"
        onPress={onPress}
      />
    )

    expect(screen.getByText('Personal Info')).toBeTruthy()
    fireEvent.press(screen.getByText('Personal Info'))
    expect(onPress).toHaveBeenCalledTimes(1)
  })

  it('hides the badge when it is zero or negative', () => {
    const { rerender, queryByText } = render(
      <ProfileMenuCell icon="cart" title="Orders" color="#000" badge={0} />
    )
    expect(queryByText('0')).toBeNull()

    rerender(<ProfileMenuCell icon="cart" title="Orders" color="#000" badge={-3} />)
    expect(queryByText('-3')).toBeNull()
  })

  it('shows the exact badge count for small numbers', () => {
    render(<ProfileMenuCell icon="cart" title="Orders" color="#000" badge={5} />)
    expect(screen.getByText('5')).toBeTruthy()
  })

  it('caps the badge label at 99+ for large counts', () => {
    render(<ProfileMenuCell icon="cart" title="Orders" color="#000" badge={128} />)
    expect(screen.getByText('99+')).toBeTruthy()
    expect(screen.queryByText('128')).toBeNull()
  })
})
