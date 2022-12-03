import { Module } from '@nestjs/common';
import { AppController } from './app.controller';
import { AppService } from './app.service';
import { BlogController } from './controller/blog/blog.controller';
import { UserController } from './controller/user/user.controller';

@Module({
  imports: [],
  controllers: [AppController, BlogController, UserController],
  providers: [AppService],
})
export class AppModule {}
