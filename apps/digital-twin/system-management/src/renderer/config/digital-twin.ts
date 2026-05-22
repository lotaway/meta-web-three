const apiPort = import.meta.env.VITE_DIGITAL_TWIN_API_PORT || '10102'
const apiHost = import.meta.env.VITE_DIGITAL_TWIN_API_HOST || 'localhost'

export const DIGITAL_TWIN_API_BASE_URL =
  import.meta.env.VITE_DIGITAL_TWIN_API_URL || `http://${apiHost}:${apiPort}`

export const DIGITAL_TWIN_WS_URL =
  import.meta.env.VITE_DIGITAL_TWIN_WS_URL ||
  `ws://${apiHost}:${apiPort}/ws/digital-twin`
