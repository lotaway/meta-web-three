import {Controller, Req, Res, Get, Post, Render, Param} from '@nestjs/common';
import {UserService} from "./user.service";
//  验证码
import svgCaptcha from "svg-captcha";
import {ControllerDto} from "./dto/user.dto";
@Controller('user')
export class UserController {

    constructor(private readonly userService: UserService) {
    }

    @Post("signIn")
    async signIn(@Param() {username, password}: ControllerDto.SignInParam, @Res() res) {
        if (!username || !password) {
            res.status = 403;
            res.message = "缺少参数";
        }
        const result = await this.userService.signIn({username, password});
        res.status = 200;
        res.data = result;
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
