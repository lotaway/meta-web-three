import {Module} from '@nestjs/common';
import {AppController} from './app.controller';
import {AppService} from './app.service';
import {UserModule} from "../user/user.module";
import {ProductModule} from "../product/product.module";
import {BlogModule} from "../blog/blog.module";

@Module({
    imports: [UserModule, ProductModule, BlogModule],
    controllers: [AppController],
    providers: [AppService],
})
export class AppModule {

}