/**
 * Centralized environment variable access with defaults.
 *
 * Every optional variable falls back to a default so that
 * `import.meta.env.XXX` can never be undefined, which would otherwise leave
 * the axios baseURL empty and break every API request.
 */

/** Default gateway address: the backend gateway listens on port 10081 locally */
export const DEFAULT_BASE_SERVER_URL = 'http://localhost:10081'

/** Default upload endpoint: routed by the gateway to the MinIO API of platform-service */
export const DEFAULT_UPLOAD_URL = '/platform-service/minio/upload'

export interface EnvVars {
  VITE_BASE_SERVER_URL?: string
  VITE_API_DOC_HOST?: string
  VITE_UPLOAD_URL?: string
  VITE_BLOCKCHAIN_ENABLED?: string
}

export interface ResolvedEnv {
  BASE_SERVER_URL: string
  API_DOC_HOST: string
  UPLOAD_URL: string
  BLOCKCHAIN_ENABLED: boolean
}

function envString(value: string | undefined, fallback: string): string {
  const trimmed = value?.trim()
  return trimmed ? trimmed : fallback
}

function envBoolean(value: string | undefined, fallback: boolean): boolean {
  const trimmed = value?.trim()
  if (!trimmed) return fallback
  return trimmed.toLowerCase() === 'true'
}

/** Resolve from explicit variables so tests never depend on an ambient .env file. */
export function resolveEnv(vars: EnvVars = {}): ResolvedEnv {
  const baseServerUrl = envString(vars.VITE_BASE_SERVER_URL, DEFAULT_BASE_SERVER_URL)
  return {
    BASE_SERVER_URL: baseServerUrl,
    API_DOC_HOST: envString(vars.VITE_API_DOC_HOST, baseServerUrl),
    UPLOAD_URL: envString(vars.VITE_UPLOAD_URL, DEFAULT_UPLOAD_URL),
    BLOCKCHAIN_ENABLED: envBoolean(vars.VITE_BLOCKCHAIN_ENABLED, false),
  }
}

const resolved = resolveEnv(import.meta.env)

/** Gateway base URL; configure a LAN address in `.env` to reach it from other devices */
export const BASE_SERVER_URL = resolved.BASE_SERVER_URL

/** OpenAPI document host; falls back to the gateway base URL */
export const API_DOC_HOST = resolved.API_DOC_HOST

/** File upload endpoint, relative to the gateway */
export const UPLOAD_URL = resolved.UPLOAD_URL

/** Blockchain feature switch, disabled by default */
export const BLOCKCHAIN_ENABLED = resolved.BLOCKCHAIN_ENABLED
