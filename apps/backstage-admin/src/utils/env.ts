/**
 * 集中读取环境变量并提供默认值。
 *
 * <p>所有可选环境变量在此处兜底，避免 `import.meta.env.XXX` 为 undefined 时
 * 出现「axios baseURL 为空导致全部 API 请求失败」这类问题。</p>
 */

/** 网关默认地址：本地开发时后端 gateway 监听 10081 端口 */
export const DEFAULT_BASE_SERVER_URL = 'http://localhost:10081'

/** 文件上传端点默认值：经网关路由到 platform-service 的 MinIO 上传接口 */
export const DEFAULT_UPLOAD_URL = '/platform-service/minio/upload'

/** 读取字符串型环境变量，为空时回退到默认值 */
function envString(value: string | undefined, fallback: string): string {
  const trimmed = value?.trim()
  return trimmed ? trimmed : fallback
}

/** 读取布尔型环境变量，仅 'true'（忽略大小写）视为开启 */
function envBoolean(value: string | undefined, fallback: boolean): boolean {
  const trimmed = value?.trim()
  if (!trimmed) return fallback
  return trimmed.toLowerCase() === 'true'
}

/**
 * 后端网关基础地址。
 * - 值为空时回退到 {@link DEFAULT_BASE_SERVER_URL}，保证 axios baseURL 始终有效
 * - 其他设备访问时在 `.env` 中配置，例如 `VITE_BASE_SERVER_URL=http://192.168.1.110:10081`
 */
export const BASE_SERVER_URL = envString(
  import.meta.env.VITE_BASE_SERVER_URL,
  DEFAULT_BASE_SERVER_URL,
)

/**
 * OpenAPI 文档地址（供 openapi-generator 拉取聚合文档）。
 * 留空时与 {@link BASE_SERVER_URL} 相同。
 */
export const API_DOC_HOST = envString(
  import.meta.env.VITE_API_DOC_HOST,
  BASE_SERVER_URL,
)

/** 文件上传端点（相对于网关） */
export const UPLOAD_URL = envString(import.meta.env.VITE_UPLOAD_URL, DEFAULT_UPLOAD_URL)

/** 区块链功能开关，默认关闭 */
export const BLOCKCHAIN_ENABLED = envBoolean(import.meta.env.VITE_BLOCKCHAIN_ENABLED, false)
