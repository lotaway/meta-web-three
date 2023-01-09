import {prismaClientProvider} from "../src/utils/connect-prisma";
// import {nanoid} from "nanoid";

const prismaClient = prismaClientProvider()

async function main() {
    try {
        /*const userCreateData = {
            // id: window.crypto.randomUUID()
            // id: nanoid()
        }
        const createUserRes = await prismaClient.user.create({
            data: userCreateData
        })
        const email = "111@qq.com"
        const createAuthorRes = await prismaClient.author.create({
            data: {
                email,
                userId: {
                    connect: {
                        id: createUserRes.id
                    }
                },
                isEnable: true
            }
        })*/
        /*const createArticleRes = await prismaClient.article.create({
            data: {
                title: "这就是文章标题",
                content: "这就是文章内容",
                state: 1,
                author: {
                    connect: {
                        // email: createAuthorRes.email
                        id: createAuthorRes.id
                    }
                }
            }
        })*/
        const createAuthorRes = {
            id: 1
        }
        //  使用了text字段类型后，无法支持create和update方法，只能直接写SQL语句
        const createArticleRes = await prismaClient.$queryRaw`insert into article (id,title,content,state,sourceName,sourceUrl,authorId) values (uuid(),'这就是文章标题1','这就是文章内容2',1,'','google.com',${createAuthorRes.id})` as { id: string }
        /*const updateArticleRes = await prismaClient.article.update({
            where: {
                id: createArticleRes.id
            },
            data: {
                author: {
                    connect: {
                        id: createAuthorRes.id
                    }
                }
            }
        })*/
        // console.log(updateArticleRes)
        // console.log(prismaClient.$queryRaw`select top(10) * from Article where createTime>now()`)
    } catch (err) {
        console.log("错误了：" + JSON.stringify(err))
    }
}

main().then(() => {
    console.log("完成")
}).catch(err => {
    prismaClient.$disconnect()
})