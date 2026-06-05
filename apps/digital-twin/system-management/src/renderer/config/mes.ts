const apiPort = import.meta.env.VITE_MES_API_PORT || '8080'
const apiHost = import.meta.env.VITE_MES_API_HOST || 'localhost'

export const MES_API_BASE_URL =
  import.meta.env.VITE_MES_API_URL || `http://${apiHost}:${apiPort}`
