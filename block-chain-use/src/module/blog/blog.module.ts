import * as nest from "@nestjs/common";
import {BlogController} from "./blog.controller";
import {BlogCommentController} from "./comment.controller";
import {CommentService} from "./comment.service";
import {DemoController} from "./demo.controller";
import {DemoService} from "./demo.service";
import { RedisService } from "../public/redis.service";

@nest.Module({
    controllers: [BlogController, BlogCommentController, DemoController],
    providers: [CommentService, DemoService, RedisService]
})
export class BlogModule {
}
