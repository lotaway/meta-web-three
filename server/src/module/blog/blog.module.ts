import {Module} from "@nestjs/common";
import {BlogController} from "./blog.controller";
import {BlogCommentController} from "./comment.controller";
import {CommentService} from "./comment.service";
import {DemoController} from "./demo.controller";

@Module({
    controllers: [BlogController, BlogCommentController, DemoController],
    providers: [CommentService]
})
export class BlogModule {
}