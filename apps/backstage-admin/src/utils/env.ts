/**
 * Centralized environment variable access with defaults.
 *
 * <p>Falls back to a default for every optional variable so that
 * `import.meta.env.XXX` can never be undefined, which would otherwise leave
 * the axios baseURL empty and break every API request.</p>
 */

/** Default gateway address: the backend gateway listens on port 10081 locally */
export const DEFAULT_BASE_SERVER_URL = 'http://localhost:10081'

/** Default upload endpoint: routed by the gateway to the MinIO API of platform-service */
export const DEFAULT_UPLOAD_URL = '/platform-service/minio/upload'

function envString(value: string | undefined, fallback: string): string {
  const trimmed = value?.trim()
  return trimmed ? trimmed : fallback
}

function envBoolean(value: string | undefined, fallback: boolean): boolean {
  const trimmed = value?.trim()
  if (!trimmed) return fallback
  return trimmed.toLowerCase() === 'true'
}

/** Gateway base URL; configure a LAN address in `.env` to reach it from other devices */
export const BASE_SERVER_URL = envString(
  import.meta.env.VITE_BASE_SERVER_URL,
  DEFAULT_BASE_SERVER_URL,
)

/** OpenAPI document host; falls back to the gateway base URL */
export const API_DOC_HOST = envString(
  import.meta.env.VITE_API_DOC_HOST,
  BASE_SERVER_URL,
)

/** File upload endpoint, relative to the gateway */
export const UPLOAD_URL = envString(import.meta.env.VITE_UPLOAD_URL, DEFAULT_UPLOAD_URL)

/** Blockchain feature switch, disabled by default */
export const BLOCKCHAIN_ENABLED = envBoolean(import.meta.env.VITE_BLOCKCHAIN_ENABLED, false)
