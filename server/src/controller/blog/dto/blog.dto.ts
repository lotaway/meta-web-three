export type BlogId = string;

export type UserId = string;

export class BlogDto {
    id: BlogId
    author: UserId
    title: string
    content: string
    createTime: Date
    updateTime: Date
    updateAuth: UserId[]
}

export type CreateBlogDto = Pick<BlogDto, "author" | "title" | "content" | "createTime">

type CommendId = number;

export class BlogComment {
    id: CommendId
    blogId: BlogId
    replyId: CommendId
    menthon: UserId[]
    content: string
    createTime: Date
    updateTime: Date
}

export type CreateBlogCommentDto = Pick<BlogComment, "id" | "blogId" | "replyId" | "menthon" | "content" | "createTime">;

export type FindBlogCommentDto = Partial<{
    commentId: CommendId
    blogId: BlogId
}>