import { ArgumentsHost, Catch, ExceptionFilter, HttpException, HttpStatus } from '@nestjs/common';
import CustomException from '../exception/service.exception';

@Catch()
export class HttpExceptionFilter<T extends CustomException.ServiceException> implements ExceptionFilter {
  catch(exception: T, host: ArgumentsHost) {
    const ctx = host.switchToHttp();
    const response = ctx.getResponse();
    const request = ctx.getRequest();
    const json = exception.toJSON();
    response.status(json.statusCode).json({
      ...json,
      path: request.url,
    });
  }
}
