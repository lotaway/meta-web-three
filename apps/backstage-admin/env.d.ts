/// <reference types="vite/client" />
/** Extends the import.meta.env type with the variables declared in .env files */
interface ImportMetaEnv {
  /**
   * Backend gateway base URL.
   * Optional: falls back to `http://localhost:10081` (see src/utils/env.ts).
   * Configure a LAN address to reach the gateway from other devices, e.g.
   * `http://192.168.1.110:10081`.
   */
  readonly VITE_BASE_SERVER_URL?: string
  /**
   * OpenAPI document host used by openapi-generator to fetch the aggregated
   * gateway document.
   * Optional: falls back to `VITE_BASE_SERVER_URL` (see src/utils/env.ts).
   */
  readonly VITE_API_DOC_HOST?: string
  /**
   * File upload endpoint, relative to the gateway.
   * Optional: falls back to `/platform-service/minio/upload` (see src/utils/env.ts).
   */
  readonly VITE_UPLOAD_URL?: string
  /**
   * Blockchain feature switch (true = shown, false = hidden).
   * Optional: falls back to false (see src/utils/env.ts).
   */
  readonly VITE_BLOCKCHAIN_ENABLED?: string
}
/** Extends the import.meta type */
interface ImportMeta {
  readonly env: ImportMetaEnv
}
