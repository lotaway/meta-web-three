import * as nest from "@nestjs/common";
import {UserController} from "./user.controller";
import {UserService} from "./user.service";
import {BlackListMiddleware} from "../../common/middleware/black-list.middleware";
import { RedisService } from "../public/redis.service";

@nest.Module({
    controllers: [UserController],
    providers: [UserService, RedisService]
})
export class UserModule implements nest.NestModule {
    configure(consumer: nest.MiddlewareConsumer) {
        consumer.apply(BlackListMiddleware).forRoutes({
            path: "user/index",
            method: nest.RequestMethod.GET
        });
        consumer.apply(BlackListMiddleware).forRoutes({
            path: "user/blog",
            method: nest.RequestMethod.GET
        });
    }
}
