import {prismaClientProvider} from "../src/utils/connect-prisma";
// import {nanoid} from "nanoid";

const prismaClient = prismaClientProvider()

async function main() {
    try {
        const userCreateData = {
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
        })
        // const createArticleRes = await prismaClient.article.create({
        //     data: {
        //         title: "这就是文章标题",
        //         content: "这就是文章内容",
        //         state: 1,
        //         author: {
        //             connect: {
        //                 // email: createAuthorRes.email
        //                 id: createAuthorRes.id
        //             }
        //         }
        //     }
        // })
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
        //  多种查询条件
        await prismaClient.article.findMany({
            skip: 20,    //  分页用，跳过多少个
            take: 10,   //  分页用，获取多少个
            where: {
                state: 1,
                title: {
                    startsWith: "web3"
                },
                // content: {
                //     startsWith: "web3"
                // }
            },
            select: {
                id: true,
                title: true,
                author: {
                    select: {
                        id: true,
                        name: true,
                        email: true
                    }
                }
            }
        })
        //  关联表的联合查询
        await prismaClient.author.findUnique({
            where: {
                email: "576696294@qq.com"
            },
            include: {
                articles: true  //  设置在author表模型的外键反关联键
            }
        })
        //  另一种关联表查询
        await prismaClient.author.findUnique({
            where: {
                email: "890@qq.com"
            }
        }).articles({
            take: 5,
            where: {
                title: {
                    contains: "科幻"
                }
            }
        })
        //  通过关联同时创建，不需要事务
        await prismaClient.user.create({
            data: {
                author: {
                    create: {
                        email: "890@qq.com"
                    }
                }
            }
        })
    } catch (err) {
        console.log("错误了：" + JSON.stringify(err))
    }
}

main().then(() => {
    console.log("完成")
}).catch(err => {
    throw err
}).finally(() => {
    prismaClient.$disconnect()
})