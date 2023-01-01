import {join} from "path";
import {NestFactory} from '@nestjs/core';
import {NestExpressApplication} from "@nestjs/platform-express";
import * as ejs from "ejs";
import * as cookieParser from "cookie-parser";
import {AppModule} from './module/app/app.module';

// import webConfig from "./config/web-config";

async function bootstrap(port: number = 30000) {
    const app = await NestFactory.create<NestExpressApplication>(AppModule);
    app.useStaticAssets(join(__dirname, "static"), {
        prefix: "/static/"
    });
    app.setBaseViewsDir(join(__dirname, "../pages"));
    app.engine('html', ejs.__express);
    app.setViewEngine("html");
    app.use(cookieParser());
    await app.listen(port);
}

bootstrap();