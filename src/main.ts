import {join} from "path";
import {NestFactory} from '@nestjs/core';
import {NestExpressApplication} from "@nestjs/platform-express";
import * as cookieParser from "cookie-parser";
import {AppModule} from './app.module';

// import webConfig from "./config/web-config";

async function bootstrap(port: number = 3000) {
    const app = await NestFactory.create<NestExpressApplication>(AppModule);
    app.useStaticAssets(join(__dirname, "static"), {
        prefix: "/static/"
    });
    app.setBaseViewsDir(join(__dirname, "../pages"));
    app.setViewEngine("pug");
    app.use(cookieParser);
    await app.listen(port);
}

bootstrap();