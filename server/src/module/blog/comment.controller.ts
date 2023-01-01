import {Controller, Get, Param, Post, Put, Delete, Body, Res, ParseUUIDPipe} from "@nestjs/common";
import {CreateBlogCommentParams, FindBlogCommentParams, CommentId} from "./dto/blog.dto";
import {CommentService} from "./comment.service";

@Controller("comment")
export class BlogCommentController {

    constructor(private readonly commentService: CommentService) {
    }

    @Get("byBlog/:blogId")
    getCommentsByBlogId(@Param() params: FindBlogCommentParams) {
        const comments = this.commentService.getCommentsByBlogId(params.blogId);
        return {
            blogId: params.blogId,
            comments
        };
    }

    @Get("byId/:commentId")
    getCommentById(@Param("commendId", new ParseUUIDPipe()) commendId: CommentId) {

    }

    @Post("byBlog")
    addBlogComment(@Body() body: CreateBlogCommentParams, @Res() res) {
        const now = new Date();
        const result = this.commentService.addBlogComment({
            ...body,
            createTime: now,
            updateTime: now
        });
        if (result) {
            res.state = 200;
        }
        return res;
    }

    @Put(":blogId/:commendId")
    updateBlogComment() {
        return "Not allow edit comment twice!";
    }

    @Delete()
    deleteBlogComment(@Param() {commentId}) {
        return "delete blog comment by commentId";
    }

}