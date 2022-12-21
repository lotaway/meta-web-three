import {Controller, Req, Res, Get, Render} from '@nestjs/common';
//  验证码
import svgCaptcha from "svg-captcha";

@Controller('user')
export class UserController {

    @Get(["", "index"])
    @Render("user/index")
    index(@Req() req, @Res() res) {
        res.cookie("timestamp", +new Date().toString());
        return {welcome: "Hello TTTTThe User.", cookies: req.cookies};
    }

    @Get(["blog", "blog/all"])
    getUserBlog(@Req() req) {
        return "Get the log in user's blogs.";
    }

}
