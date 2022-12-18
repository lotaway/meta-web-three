import {Controller, Request, Get, Render} from '@nestjs/common';
//  验证码
import svgCaptcha from "svg-captcha";

@Controller('user')
export class UserController {

    @Get(["", "index"])
    @Render("user/index")
    index(@Request() request) {
        return {welcome: "Hello TTTTThe User.", cookies: request.cookies};
    }

}
