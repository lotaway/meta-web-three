import Redis from "ioredis"

let client: Redis = null;

export function redisClientProvider(): typeof client {
    if (client !== null) {
        return client;
    }
    client = new Redis({
        host: process.env.REDIS_HOST ?? "127.0.0.1",
        port: Number(process.env.REIDS_PORT ?? 6379),
        password: process.env.REDIS_PASSWORD,
    });
    client.on("reconnecting", msg => {
        console.log("redis trying reconnect")
    })
    client.on("error", err => {
        console.log("redis has problem: " + JSON.stringify(err));
    });
    return client;
}
