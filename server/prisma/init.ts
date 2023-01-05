import {prismaClientProvider} from "../src/utils/connect-prisma";

const prismaClient = prismaClientProvider()

async function main() {
    try {
        await prismaClient.user.create({
            data: {
                email: "576696294@qq.com"
            }
        })
    } catch (err) {
        console.log("错误了：" + JSON.stringify(err))
    }
}

main().then(() => {
    console.log("完成")
}).catch(err => {
    prismaClient.$disconnect()
})