import {HttpException, Injectable, NestMiddleware} from "@nestjs/common";
import {Request, Response, NextFunction} from "express";

const allUserComeFromDB = [];

@Injectable()
export class ValidUserMiddleware implements NestMiddleware {

    use(req: Request, res: Response, next: NextFunction) {
        const userId = req.params.userId;
        const isExist = allUserComeFromDB.some(item => item.id === userId);
        if (!isExist) {
            throw new HttpException("Didn't find match user.", 400);
        }
        next();
    }

}