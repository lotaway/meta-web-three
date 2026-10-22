/// <reference types="vite/client" />
/** 扩展import.meta.env对象类型，配置.env文件中的环境变量 */
interface ImportMetaEnv {
  /**
   * 后端网关基础地址。
   * 可选：留空时回退到 `http://localhost:10081`（见 src/utils/env.ts）。
   * 其他设备访问时配置为局域网地址，例如 `http://192.168.1.110:10081`。
   */
  readonly VITE_BASE_SERVER_URL?: string
  /**
   * OpenAPI 文档地址（供 openapi-generator 拉取聚合文档）。
   * 可选：留空时与 `VITE_BASE_SERVER_URL` 相同（见 src/utils/env.ts）。
   */
  readonly VITE_API_DOC_HOST?: string
  /**
   * 文件上传端点（相对网关地址）。
   * 可选：留空时回退到 `/platform-service/minio/upload`（见 src/utils/env.ts）。
   */
  readonly VITE_UPLOAD_URL?: string
  /**
   * 区块链功能开关（true=显示, false=隐藏）。
   * 可选：留空时回退到 false（见 src/utils/env.ts）。
   */
  readonly VITE_BLOCKCHAIN_ENABLED?: string
}
/** 扩展import.meta对象类型 */
interface ImportMeta {
  readonly env: ImportMetaEnv
}
