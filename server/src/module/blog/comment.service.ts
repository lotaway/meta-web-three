import {Injectable} from "@nestjs/common";
import {BlogId, CommentId, BlogComment} from "./dto/blog.dto";

// import {nanoid} from "nanoid";

@Injectable()
export class CommentService {
    comments: BlogComment[] = []

    getCommentsByBlogId(blogId: BlogId) {
        return this.comments.filter(item => item.blogId === blogId);
    }

    getCommentById(commendId: CommentId) {
        return this.comments.find(item => item.id === commendId);
    }

    addBlogComment(payload: Omit<BlogComment, "id">) {
        const finalData = {
            // id: nanoid(),
            id: crypto.randomUUID(),
            ...payload
        };
        this.comments.push(finalData);
        return true;
    }

}