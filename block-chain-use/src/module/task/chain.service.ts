import { Injectable } from "@nestjs/common";
import { Cron } from "@nestjs/schedule";

@Injectable()
export class ChainService {

    @Cron('* */5 * * * *')
    task() {
        console.log('This task runs every minute');
    }
}