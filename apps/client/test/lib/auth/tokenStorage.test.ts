import AsyncStorage from '@react-native-async-storage/async-storage'
import { clearToken, DEFAULT_TOKEN_TTL_MS, getToken, setToken } from '@/lib/auth/tokenStorage'

const TOKEN_KEY = '@meta_web_three:auth_token'
const EXPIRES_AT_KEY = '@meta_web_three:auth_token_expires_at'

describe('tokenStorage', () => {
  beforeEach(() => {
    jest.clearAllMocks()
    ;(AsyncStorage.getItem as jest.Mock).mockImplementation((key: string) =>
      Promise.resolve(key === TOKEN_KEY ? 'stored-token' : null),
    )
    ;(AsyncStorage.setItem as jest.Mock).mockResolvedValue(undefined)
    ;(AsyncStorage.removeItem as jest.Mock).mockResolvedValue(undefined)
  })

  it('reads a valid token', async () => {
    await expect(getToken()).resolves.toBe('stored-token')
    expect(AsyncStorage.getItem).toHaveBeenCalledWith(TOKEN_KEY)
  })

  it('clears and returns null when the token is expired', async () => {
    const past = String(Date.now() - 1000)
    ;(AsyncStorage.getItem as jest.Mock).mockImplementation((key: string) =>
      Promise.resolve(key === TOKEN_KEY ? 'stored-token' : past),
    )

    await expect(getToken()).resolves.toBeNull()
    expect(AsyncStorage.removeItem).toHaveBeenCalledWith(TOKEN_KEY)
    expect(AsyncStorage.removeItem).toHaveBeenCalledWith(EXPIRES_AT_KEY)
  })

  it('persists the token with an expiry using the default TTL', async () => {
    const now = Date.now()
    jest.spyOn(Date, 'now').mockReturnValue(now)
    await setToken('new-token')

    expect(AsyncStorage.setItem).toHaveBeenCalledWith(TOKEN_KEY, 'new-token')
    expect(AsyncStorage.setItem).toHaveBeenCalledWith(EXPIRES_AT_KEY, String(now + DEFAULT_TOKEN_TTL_MS))
    jest.restoreAllMocks()
  })

  it('honours a custom TTL', async () => {
    const now = Date.now()
    jest.spyOn(Date, 'now').mockReturnValue(now)
    await setToken('new-token', 1000)

    expect(AsyncStorage.setItem).toHaveBeenCalledWith(EXPIRES_AT_KEY, String(now + 1000))
    jest.restoreAllMocks()
  })

  it('removes both keys on clear', async () => {
    await clearToken()
    expect(AsyncStorage.removeItem).toHaveBeenCalledWith(TOKEN_KEY)
    expect(AsyncStorage.removeItem).toHaveBeenCalledWith(EXPIRES_AT_KEY)
  })
})
