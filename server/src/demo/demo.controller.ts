import {Controller, Get, Render} from "@nestjs/common";

enum Router {
    all = "all"
}

@Controller("demo")
export class DemoController {

    @Get(["", Router.all])
    @Render(`demo/${Router.all}`)
    webComponent() {
        return {};
    }

}