import {Controller, Get, Param, Render} from "@nestjs/common";
import {DemoService} from "./demo.service";

enum Router {
    all = "all"
}

@Controller("demo")
export class DemoController {

    constructor(private readonly demoService: DemoService) {
    }

    @Get(["", Router.all])
    @Render(`demo/${Router.all}`)
    webComponent() {
        return {};
    }

    @Get("user/all")
    async getAllUsers() {
        return await this.demoService.getAllUsers()
    }

    @Get("file/:fileName")
    async getMarkDownFile(@Param("fileName") fileName) {
        return await this.demoService.getFileByName(fileName)
    }

}