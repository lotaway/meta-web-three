import {Controller, Get, Render} from "@nestjs/common";

enum Router {
    WebComponent = "web-component"
}

@Controller("demo")
export class DemoController {

    @Get(["", Router.WebComponent])
    @Render(`demo/${Router.WebComponent}`)
    webComponent() {
        return {};
    }

}