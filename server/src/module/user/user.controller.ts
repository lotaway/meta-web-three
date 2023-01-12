import {Controller, Req, Res, Get, Post, Render, Param} from '@nestjs/common';
import {UserService} from "./user.service";
//  验证码
import svgCaptcha from "svg-captcha";
import {SignInParam} from "./dto/user.dto";

@Controller('user')
export class UserController {

    constructor(private readonly userService: UserService) {
    }

    @Post("signIn")
    signIn(@Param() {account, password}: SignInParam) {
        if (!account || !password) {
            return {

            }
        }
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
