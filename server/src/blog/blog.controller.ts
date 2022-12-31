import {Controller, Get, Post, Put, Delete, Redirect, Param, Query, Body} from '@nestjs/common';
import {BlogId, CreateBlog} from "./dto/blog.dto";
import {getRedisClient} from "../utils/connect-redis";

function defaultParam(defaultParams?: any[]) {
    return (target: any, propName: string, descriptor: PropertyDescriptor) => {
        const originFn = descriptor.value;

        descriptor.value = (...realParams: any[]) => {
            console.log("params: " + JSON.stringify(realParams));
            // return Reflect.apply(target, propName, realParams);
            return originFn.call(this, ...realParams);
        }
    };
}

interface BlogListQueries extends Object {
    p?: number
}

@Controller('blog')
export class BlogController {

    redisClient = getRedisClient()

    @Get(["", "index"])
    // @Redirect("/blog/list")
    async blogIndex() {
        return await new Promise((resolve) => {
            setTimeout(() => {
                resolve("Blog Index OK.")
            }, 1000);
        });
    }

    @Get("list")
    // @defaultParam()
    async blogList(@Query() query: BlogListQueries = {}) {
        try {
            //  todo 抓取多个网站新闻内容并显示在这里，缓存相应的数据，之后应当使用mysql规则存储每日抓取的数据
            await this.redisClient.set("blog", "active")
            return await this.redisClient.get("blog")
        } catch (err) {
            return "blog redis error: " + JSON.stringify(err)
        }
    }

    @Get("static/:blogNumber")
    getStaticDetail(@Param("blogNumber") blogNumber: string) {
        return `Get the blog by id ${blogNumber}`;
    }

    @Get("detail")
    getBlogDetail(@Query("id") blogId: BlogId) {
        return `get blog by blogId is: ${blogId}`;
    }

    @Post()
    addBlog(@Body() {author}: CreateBlog) {
        return `Add a new blog with author ${author}`;
    }

    @Put(":blogId")
    updateBlogDetail(@Param("blogId") blogId: BlogId) {
        return `Start edit blog with id ${blogId}`;
    }

    @Delete(":blogId")
    deleteBlogDetail(@Param("blogId") blogId: BlogId) {
        return `Gone to delete blog with id ${blogId}`;
    }
}
