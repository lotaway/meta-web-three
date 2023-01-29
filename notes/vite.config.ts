import {defineConfig} from 'vite'
import vue from '@vitejs/plugin-vue'
import electron from "vite-plugin-electron"
import electronRenderer from "vite-plugin-electron-renderer"
import path from "path"

// https://vitejs.dev/config/
export default defineConfig({
    plugins: [
        vue(),
        electron([{
            entry: "./src/main/desktop-main.js"
        }]),
        electronRenderer({
            nodeIntegration: true
        })
    ],
    base: "./", //  定义基础路径为相对路径，否则vite生成的路径为相对网站根目录的`/xxx`，而electron需要本地路径`./xxx`
    server: {
        port: 30002,
        // open: true
    },
    build: {
        rollupOptions: {
            // input: path.resolve(__dirname, "./index.html")
        }
    }
})
