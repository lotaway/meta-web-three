import * as redis from "redis";

const redisConfig = {
    host: "http://127.0.0.1",
    port: 6379
}
//  todo 添加了redis地址会提示Invalid URL
// const redisURL = `redis://${redisConfig.port}:${redisConfig.host}`
const redisURL = ""
let client: ReturnType<typeof redis.createClient> = null

export function redisClientProvider(): typeof client {
    if (client === null) {
        client = redis.createClient({
            url: redisURL
        })
        client.connect()
        client.on("error", err => {
            console.log("redis has problem: " + JSON.stringify(err))
        })
    }
    return client
}