import {MiddlewareConsumer, Module, NestModule, RequestMethod} from "@nestjs/common";
import {UserController} from "./user.controller";
import {UserService} from "./user.service";
import {ValidUserMiddleware} from "../../common/middleware/valid-user.middleware";

@Module({
    controllers: [UserController],
    providers: [UserService]
})
export class UserModule implements NestModule {
    configure(consumer: MiddlewareConsumer) {
        consumer.apply(ValidUserMiddleware).forRoutes({
            path: "user/index",
            method: RequestMethod.GET
        });
        consumer.apply(ValidUserMiddleware).forRoutes({
            path: "user/blog",
            method: RequestMethod.GET
        });
    }
}
