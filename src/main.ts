import {join} from "path";
import {NestFactory} from '@nestjs/core';
import {AppModule} from './app.module';
import {NestExpressApplication} from "@nestjs/platform-express";
// import webConfig from "./config/web-config";

async function bootstrap(port: number = 3000) {
    const app = await NestFactory.create<NestExpressApplication>(AppModule);
    app.useStaticAssets(join(__dirname, "static"), {
        prefix: "/static/"
    });
    app.setBaseViewsDir(join(__dirname, "../pages"));
    app.setViewEngine("pug");
    await app.listen(port);
}

bootstrap();