import { HttpException, HttpStatus } from "@nestjs/common";

namespace CustomException {

    export enum ErrorCode {
        SERVICE_ERROR = 0,
        USER_NOT_FOUND = 10000,
        USER_NOT_AUTHENTICATED = 10002,
        USER_AUTHROZATION_VALID = 10004,
    }

    export class ServiceException extends HttpException {
        constructor(private readonly errorCode: ErrorCode, message: string, status: number) {
            super(message, status);
        }

        static create(errorCode = ErrorCode.SERVICE_ERROR): ServiceException {
            return new ServiceException(errorCode, "OK", HttpStatus.OK);
        }

        getErrorCode(): ErrorCode {
            return this.errorCode
        }

        toJSON() {
            return {
                errorCode: this.errorCode,
                message: this.message,
                statusCode: this.getStatus(),
                timestamp: new Date().toISOString(),
            }
        }
    }

}

export default CustomException