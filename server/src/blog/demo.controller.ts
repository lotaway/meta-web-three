import {join} from "path";
import {createReadStream} from "fs";
import {Controller, Get, Render} from "@nestjs/common";
import {getRedisClient} from "../utils/connect-redis";

enum Router {
    all = "all"
}

@Controller("demo")
export class DemoController {

    redisClient = getRedisClient()

    @Get(["", Router.all])
    @Render(`demo/${Router.all}`)
    webComponent() {
        return {};
    }

    @Get("file")
    async getMarkDownFile() {
        if (false) {
            try {
                //  todo 抓取多个网站新闻内容并显示在这里，缓存相应的数据，之后应当使用mysql规则存储每日抓取的数据
                await this.redisClient.set("blog", "contentCache")
                return await this.redisClient.get("blog")
            } catch (err) {
                return "blog redis error: " + JSON.stringify(err)
            }
        }
        const readStream = createReadStream(join(__dirname, "./demo.controller.js"))
        const result = await new Promise((resolve, reject) => {
            readStream.on("error", err => {
                console.log("read file error: " + JSON.stringify(err))
                reject(err)
            })
            let fileChunkArr = []
            readStream.on("data", chunk => {
                fileChunkArr.push(chunk)
            })
            readStream.on("end", () => {
                const fileBuffer = fileChunkArr.join("")
                resolve(fileBuffer.toString())
            })
        })
        return result
    }

}