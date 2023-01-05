import {PrismaClient} from "@prisma/client";

let client: InstanceType<typeof PrismaClient> = null

export function prismaClientProvider() {
    if (!client) client = new PrismaClient()
    return client
}