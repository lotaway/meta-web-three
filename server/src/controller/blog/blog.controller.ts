import {Controller, Get, Redirect, Param, Query} from '@nestjs/common';

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

interface BlogListParam extends Object {
    p?: number
}

@Controller('blog')
export class BlogController {
    @Get(["", "index"])
    // @Redirect("/blog/list")
    async blogIndex() {
        return await new Promise((resolve) => {
            setTimeout(() => {
                resolve("Blog Index OK.")
            }, 1000);
        });
    }

    @Get("list/:p?")
    // @defaultParam()
    blogList(@Param() params: BlogListParam = {}) {
        return `p is: ${params.p}`;
    }

    @Get(":blogNumber")
    blogStaticDetail(@Param("blogNumber") blogNumber: number) {

    }

    @Get("detail")
    blogDetail(@Query("id") blogId: number) {
        return `blogId is: ${blogId}`;
    }
}
