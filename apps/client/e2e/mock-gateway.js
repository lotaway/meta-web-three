// Lightweight mock gateway for client e2e tests.
//
// The app's EXPO_PUBLIC_BACK_API_HOST points at a gateway (default 10081).
// In e2e/CI there is no real backend, so pending API calls keep the app busy
// forever and Detox's synchronization (isReady) times out. This server returns
// empty, well-formed ApiResponse payloads so the app settles to idle.
//
// Usage:
//   HOST=192.168.1.110 PORT=10081 node e2e/mock-gateway.js
// The host must match the app's configured API host (EXPO_PUBLIC_BACK_API_HOST).
const http = require('http')

const HOST = process.env.MOCK_GATEWAY_HOST || '127.0.0.1'
const PORT = Number(process.env.MOCK_GATEWAY_PORT || 10081)

const server = http.createServer((req, res) => {
  res.setHeader('Content-Type', 'application/json; charset=utf-8')
  res.setHeader('Access-Control-Allow-Origin', '*')
  res.setHeader('Access-Control-Allow-Headers', '*')
  res.setHeader('Access-Control-Allow-Methods', 'GET,POST,PUT,DELETE,OPTIONS')

  if (req.method === 'OPTIONS') {
    res.writeHead(200)
    res.end()
    return
  }

  req.resume()

  let data = []
  const url = req.url || ''
  if (/stats|summary|current|detail|info|profile|config/.test(url)) data = {}

  res.writeHead(200)
  res.end(JSON.stringify({ code: '0000', message: 'success', data }))
})

server.listen(PORT, HOST, () => {
  console.log(`mock gateway listening on http://${HOST}:${PORT}`)
})
