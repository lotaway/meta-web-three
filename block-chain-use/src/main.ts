import {join} from "path";
import {NestFactory, Reflector} from '@nestjs/core';
import {NestExpressApplication} from "@nestjs/platform-express";
import * as ejs from "ejs";
import * as cookieParser from "cookie-parser";
import setting from "./config/settings";
import {AppModule} from './module/app/app.module';
import { ForbittenInterceptor } from "./common/interceptor/forbitten.interceptor";
import { AuthrozationGuard } from "./common/guard/authrozation.guard";

// import webConfig from "./config/web-config";

async function bootstrap(port: number = 30001) {
    const app = await NestFactory.create<NestExpressApplication>(AppModule);
    app.useStaticAssets(join(__dirname, "static"), {
        prefix: "/static/"
    });
    app.setBaseViewsDir(join(__dirname, "../pages"));
    app.engine('html', ejs.__express);
    app.setViewEngine("html");
    app.use(cookieParser());
    app.useGlobalInterceptors(new ForbittenInterceptor(new Reflector()));
    app.useGlobalGuards(new AuthrozationGuard(new Reflector()));
    return await app.listen(port);
}

bootstrap(setting.PORT).then(server => {
    console.log(`run in ${JSON.stringify(server.address())}`)
});
