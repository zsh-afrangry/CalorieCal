import { defineConfig } from "vite";
import vue from "@vitejs/plugin-vue";
import basicSsl from '@vitejs/plugin-basic-ssl';
export default defineConfig({
    plugins: [
        vue(),
        basicSsl() // 启用基础 SSL
    ],
    server: {
        host: '0.0.0.0', // 允许局域网访问
        cors: true, // 允许跨域
        // 如果你有后端API，可能还需要配置代理来解决前端请求后端的跨域问题
        // proxy: {
        //   '/api': {
        //     target: 'http://localhost:8000', // 你的后端地址
        //     changeOrigin: true,
        //     rewrite: (path) => path.replace(/^\/api/, '')
        //   }
        // }
    }
});
