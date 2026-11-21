import AsyncStorage from '@react-native-async-storage/async-storage'

const TOKEN_KEY = '@meta_web_three:auth_token'
const EXPIRES_AT_KEY = '@meta_web_three:auth_token_expires_at'
export const DEFAULT_TOKEN_TTL_MS = 24 * 60 * 60 * 1000

export async function getToken(): Promise<string | null> {
  const [token, expiresAt] = await Promise.all([
    AsyncStorage.getItem(TOKEN_KEY),
    AsyncStorage.getItem(EXPIRES_AT_KEY),
  ])
  if (!token) return null
  if (expiresAt && Date.now() >= Number(expiresAt)) {
    await clearToken()
    return null
  }
  return token
}

export async function setToken(token: string, ttlMs: number = DEFAULT_TOKEN_TTL_MS): Promise<void> {
  const expiresAt = Date.now() + ttlMs
  await Promise.all([
    AsyncStorage.setItem(TOKEN_KEY, token),
    AsyncStorage.setItem(EXPIRES_AT_KEY, String(expiresAt)),
  ])
}

export async function clearToken(): Promise<void> {
  await Promise.all([
    AsyncStorage.removeItem(TOKEN_KEY),
    AsyncStorage.removeItem(EXPIRES_AT_KEY),
  ])
}
