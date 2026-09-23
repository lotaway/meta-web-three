import Constants from 'expo-constants'
import { Platform } from 'react-native'

/**
 * Backend gateway port (gateway service, see scripts/server-services-registry.sh).
 */
export const DEFAULT_API_PORT = 10081

/**
 * Resolve the dev machine's host reachable from the app.
 *
 * - Explicit override first: EXPO_PUBLIC_BACK_API_HOST (Metro inlines EXPO_PUBLIC_*)
 *   → NEXT_PUBLIC_BACK_API_HOST (legacy, used by web/build tooling).
 * - Otherwise derive the host from the Metro dev server the device/emulator is
 *   connected to (`Constants.expoConfig.hostUri`, e.g. "192.168.1.5:8081").
 * - Fallback per platform: Android emulator must use 10.0.2.2 to reach the host;
 *   iOS simulator can use localhost.
 */
export function getDevHost(): string {
  const hostUri = Constants.expoConfig?.hostUri
  if (hostUri) {
    return hostUri.split(':')[0]
  }
  if (Platform.OS === 'android') {
    return '10.0.2.2'
  }
  return 'localhost'
}

/**
 * Base URL of the backend gateway for the current runtime.
 * Trailing slashes are stripped.
 */
export function getApiBaseUrl(port: number = DEFAULT_API_PORT): string {
  const explicit =
    process.env.EXPO_PUBLIC_BACK_API_HOST ??
    process.env.NEXT_PUBLIC_BACK_API_HOST
  if (explicit) {
    return explicit.replace(/\/+$/, '')
  }
  return `http://${getDevHost()}:${port}`
}