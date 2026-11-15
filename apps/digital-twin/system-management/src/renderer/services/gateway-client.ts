import type { AxiosInstance } from 'axios'
import {
  DIGITAL_TWIN_WS_PATH,
  FACTORY_SERVICE_BASE_URL,
  GATEWAY_BASE_URL,
} from '../config/gateway'

const TOKEN_MISSING_MESSAGE =
  '未配置 GATEWAY_TOKEN：请在 .env 中写入平台签发的 JWT，所有 server/ 接口必须经 gateway 校验'

/**
 * The JWT is owned by the Electron main process (dotenv), not baked into the
 * renderer bundle. It is exposed through a single named preload method.
 */
export async function getGatewayToken(): Promise<string> {
  const token = await window.desktop?.getGatewayToken?.()
  if (typeof token !== 'string' || !token.trim()) {
    throw new Error(TOKEN_MISSING_MESSAGE)
  }
  return token.trim()
}

/** fetch() wrapper for every cross-service REST call. */
export async function gatewayFetch(path: string, init: RequestInit = {}): Promise<Response> {
  const token = await getGatewayToken()
  const headers = new Headers(init.headers)
  headers.set('Authorization', `Bearer ${token}`)
  if (init.body !== undefined && !headers.has('Content-Type')) {
    headers.set('Content-Type', 'application/json')
  }
  return fetch(`${FACTORY_SERVICE_BASE_URL}${path}`, { ...init, headers })
}

/** Attach the gateway JWT to an axios client created in the generated layer. */
export function installGatewayAuth(client: AxiosInstance): void {
  client.interceptors.request.use(async (config) => {
    const token = await getGatewayToken()
    config.headers.set('Authorization', `Bearer ${token}`)
    return config
  })
}

/**
 * Browsers cannot set headers on a WebSocket handshake, so the JWT is passed
 * as a ?token= query parameter. UserAuthFilter validates it and removes the
 * parameter before the request is proxied downstream.
 */
export async function buildGatewayWsUrl(path: string = DIGITAL_TWIN_WS_PATH): Promise<string> {
  const token = await getGatewayToken()
  const wsBase = GATEWAY_BASE_URL.replace(/^http/, 'ws')
  const url = new URL(path, wsBase)
  url.searchParams.set('token', token)
  return url.toString()
}
