import {Module} from '@nestjs/common';
import {AppController} from './app.controller';
import {AppService} from './app.service';
import {AnalysisController} from '../controller/analysis/analysis.controller';
import {BlogController} from '../controller/blog/blog.controller';
import {BlogCommentController} from '../controller/blog/comment.controller';
import {DemoController} from '../controller/demo/demo.controller';
import {UserController} from '../controller/user/user.controller';

@Module({
    imports: [],
    controllers: [AppController, AnalysisController, BlogController, BlogCommentController, UserController, DemoController],
    providers: [AppService],
})
export class AppModule {
}
