import {Controller, Get} from "@nestjs/common";

@Controller("analysis")
export class AnalysisController {

    @Get(["blog", "blog/all"])
    getCreatorBlogAnalysis() {
        return "get creator blog analysis.";
    }

}