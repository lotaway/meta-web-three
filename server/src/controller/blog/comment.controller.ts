import {Controller, Get, Param, Post, Put, Delete, Body} from "@nestjs/common";
import {BlogId, CreateBlogCommentDto, FindBlogCommentDto} from "./dto/blog.dto";

@Controller("comment")
export class BlogCommentController {

    @Get(":blogId")
    getBlogComment(@Param() params: FindBlogCommentDto) {
        return `Get blog comment by blog id ${params.blogId}`;
    }

    @Post(":blogId")
    addBlogComment(@Body() body: CreateBlogCommentDto) {
        return "add blog comment, should base on user auth.";
    }

    @Put(":blogId/:commendId")
    updateBlogComment() {
        return "Not allow edit comment twice!";
    }

    @Delete()
    deleteBlogComment(@Param() {commentId}) {
        return ""
    }

}