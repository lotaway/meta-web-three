import {Controller, Get, Res} from '@nestjs/common';
import {AppService} from './app.service';

@Controller()
export class AppController {
    constructor(private readonly appService: AppService) {
    }

    @Get()
    getHello(@Res() res): string {
        res.cookie("timestamp", +new Date().toString());
        return this.appService.getHello();
    }
}
