import * as nest from "@nestjs/common";
import {Request, Response, NextFunction} from "express";
import CustomException from "../exception/service.exception";

const blackList = [];

@nest.Injectable()
export class BlackListMiddleware implements nest.NestMiddleware {

    use(req: Request, res: Response, next: NextFunction) {
        const userId = req.params.userId;
        const isExist = blackList.find(item => item.id === userId);
        if (isExist) {
            throw CustomException.ServiceException.create();
        }
        next();
        console.log("Passed black list middleware")
    }

}
