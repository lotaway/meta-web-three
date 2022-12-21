import {UserId} from "../../user/dto/user.dto";

export type BlogId = string;

export class Blog {
    id: BlogId
    author: UserId
    title: string
    content: string
    createTime: Date
    updateTime: Date
    updateAuth: UserId[]
}

export type CreateBlog = Pick<Blog, "author" | "title" | "content" | "createTime">

export type CommentId = string

export class BlogComment {
    id: CommentId
    blogId: BlogId
    replyId: CommentId
    mentioned: UserId[]
    content: string
    createTime: Date
    updateTime: Date
}

export type CreateBlogCommentParams = Omit<BlogComment, "id" | "createTime" | "updateTime">;

export type FindBlogCommentParams = {
    blogId: BlogId
}