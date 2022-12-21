import {Module} from "@nestjs/common";
import {BlogController} from "./blog.controller";
import {BlogCommentController} from "./comment.controller";
import {CommentService} from "./comment.service";

@Module({
    controllers: [BlogController, BlogCommentController],
    providers: [CommentService]
})
export class BlogModule {
}