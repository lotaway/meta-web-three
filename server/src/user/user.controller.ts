import {Controller, Req, Res, Get, Render} from '@nestjs/common';
import {UserService} from "./user.service";
//  验证码
import svgCaptcha from "svg-captcha";

@Controller('user')
export class UserController {

    constructor(private readonly userService: UserService) {
    }

    @Get(["", "index"])
    @Render("user/index")
    index(@Req() req, @Res() res) {
        res.cookie("timestamp", +new Date().toString());
        return {welcome: "Hello TTTTThe User.", cookies: req.cookies};
    }

    @Get(["blog", "blog/all"])
    getUserBlog(@Req() req) {
        const blogCount = this.userService.getUserBlogCount();
        return {
            blog: {
                tip: `Get the log in user's blogs, you already publish ${blogCount} blogs!`,
                count: blogCount
            }
        };
    }

}