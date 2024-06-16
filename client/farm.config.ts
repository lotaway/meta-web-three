import {defineConfig} from '@farmfe/core'
import farmJsPluginSvgr from "@farmfe/js-plugin-svgr"
import {resolve} from "node:path"
import {readFileSync} from 'node:fs'

const nginxConfig = readFileSync(resolve(__dirname, 'public/nginx.conf'), 'utf8')
const locationRegex = /location\s+\^~\s+\/([\w\/]+)\s+\{[\s\S]*?proxy_pass\s+(http[s]?:\/\/[^;]+);[\s\S]*?\}/g
let proxy: Record<string, ProxyOptions> = {}
let match
while ((match = locationRegex.exec(nginxConfig)) !== null) {
    // 获取路径和目标URL
    const matchPrefix = `/${match[1]}`
    const target = match[2]
    // 添加到结果对象
    proxy[matchPrefix] = {
        target,
        changeOrigin: true,
        rewrite: (path: string) => path.replace(new RegExp(`^${matchPrefix}`), ''),
    }
}
// console.log(proxy)
// Object.values(proxy).forEach(item => console.log(item.rewrite?.toString()))

// https://www.farmfe.org/docs/quick-start
export default defineConfig({
    plugins: [
        '@farmfe/plugin-react',
        '@farmfe/plugin-sass',
        // "vite-plugin-wasm",
        farmJsPluginSvgr(),
        {
            name: 'file-loader',
            // apply: 'build',
            // enforce: 'pre',
            transform: {
                filters: {
                    moduleTypes: [".wasm"],
                },
                executor(code, id) {
                    if (id.endsWith('.wasm')) {
                        return Promise.resolve({
                            code: `export default ${JSON.stringify(id)};`,
                            map: null
                        })
                    }
                    return Promise.resolve(null)
                }
            },
        }
    ],
    server: {
        port: 30001,
        proxy,
    }
});