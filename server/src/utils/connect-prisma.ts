import {PrismaClient} from "@prisma/client";

let client = null

export function prismaClientProvider() {
    if (!client) {
        client = new PrismaClient()
    }
    return client
}