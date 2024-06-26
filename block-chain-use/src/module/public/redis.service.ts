import * as nest from '@nestjs/common';
import Redis from 'ioredis';

@nest.Injectable()
export class RedisService implements nest.OnModuleInit, nest.OnModuleDestroy {
  private client: Redis;

  onModuleInit() {
    this.client = new Redis({
        host: process.env.REDIS_HOST ?? "127.0.0.1",
        port: Number(process.env.REIDS_PORT ?? 6379),
        password: process.env.REDIS_PASSWORD,
    });
    this.client.on("reconnecting", msg => {
        console.log("redis trying reconnect")
    })
    this.client.on("error", err => {
        console.log("redis has problem: " + JSON.stringify(err));
    });
  }

  onModuleDestroy() {
    this.client.disconnect();
  }

  async getToken(userId: string): Promise<string | null> {
    return await this.client.get(userId);
  }
}
