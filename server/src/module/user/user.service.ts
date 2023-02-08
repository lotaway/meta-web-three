import {Injectable} from "@nestjs/common";
import {prismaClientProvider} from "../../utils/connect-prisma";

interface User extends Object {

}

type Provider = ReturnType<typeof prismaClientProvider>;

interface CreateUserParams {
    email: string
    password: string
}

interface CreateUserResult {

}

interface GetUserByIdParams {
    id: number
}

interface SignInParams extends Object {
    username: string
    password: string
}

interface SignInResult {

}

abstract class Service {
    protected readonly prismaClient: Provider;

    abstract createUser(params: CreateUserParams): Promise<CreateUserResult>

    abstract signIn(params: SignInParams): Promise<SignInResult>

    abstract getUserById<Result = unknown>(options?: GetUserByIdParams): Promise<Result | User>;
}

@Injectable()
export class UserService extends Service {
    protected readonly prismaClient = prismaClientProvider();

    override async createUser({email, password}) {
        return await this.prismaClient.user.create({
            data: {
                email,
                password
            }
        });
    }

    override async signIn({username, password}: SignInParams) {
        return await this.prismaClient.user.findMany({
            where: {
                email: username,
                password
            }
        });
    }

    override async getUserById({id, password}) {
        return await this.prismaClient.user.findUnique({
            where: {
                id
            }
        });
    }

    addFollower() {

    }

    getFollowers() {
        return [];
    }

    getUserBlogCount() {
        return [];
    }
}
