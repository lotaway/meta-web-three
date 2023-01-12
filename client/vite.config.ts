import {defineConfig} from 'vite'
import react from '@vitejs/plugin-react';

// https://vitejs.dev/config/
export default defineConfig({
    plugins: [
        react()
    ],
    //  配置前端服务地址和端口
    server: {
        host: '0.0.0.0',
        port: 30001,
        //  是否开启https
        // https: true,
        // 设置反向代理，跨域
        proxy: {
            '/api': {
                // 后台地址
                target: 'http://localhost:30000/',
                rewrite: path => path.replace(/^\/api/, '')
            }
        }
    }
})
