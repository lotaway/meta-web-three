import {createClient} from "redis";

const redisConfig = {
    host: "http://127.0.0.1",
    port: 6379
}
let client = null

export function getRedisClient() {
    if (!client) {
        client = createClient()
        client.connect(redisConfig.port, redisConfig.host)
        client.on("error", err => {
            console.log("redis has problem: " + JSON.stringify(err))
        })
    }
    return client
}