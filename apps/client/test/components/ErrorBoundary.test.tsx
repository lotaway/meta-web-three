import { fireEvent, render, screen } from '@testing-library/react-native'
import { Text } from 'react-native'
import ErrorBoundary from '@/components/ErrorBoundary'

let shouldThrow = true

function MaybeBoom() {
  if (shouldThrow) {
    throw new Error('render exploded')
  }
  return <Text>recovered</Text>
}

describe('components/ErrorBoundary', () => {
  let consoleError: jest.SpyInstance

  beforeEach(() => {
    shouldThrow = true
    consoleError = jest.spyOn(console, 'error').mockImplementation(() => {})
  })

  afterEach(() => {
    consoleError.mockRestore()
  })

  it('renders its children when no error occurs', () => {
    render(
      <ErrorBoundary>
        <Text>healthy child</Text>
      </ErrorBoundary>,
    )
    expect(screen.getByText('healthy child')).toBeTruthy()
  })

  it('shows a visible fallback with an alert role when a child throws', () => {
    render(
      <ErrorBoundary>
        <MaybeBoom />
      </ErrorBoundary>,
    )
    expect(screen.getByText('Something went wrong')).toBeTruthy()
    expect(
      screen.getByText('The app hit an unexpected error. Please try again.'),
    ).toBeTruthy()
    expect(screen.UNSAFE_getByProps({ accessibilityRole: 'alert' })).toBeTruthy()
  })

  it('retry clears the error state and re-renders children', () => {
    render(
      <ErrorBoundary>
        <MaybeBoom />
      </ErrorBoundary>,
    )
    shouldThrow = false
    fireEvent.press(screen.getByText('Try again'))
    expect(screen.getByText('recovered')).toBeTruthy()
  })
})
