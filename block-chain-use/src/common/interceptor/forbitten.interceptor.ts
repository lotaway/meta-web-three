import { CallHandler, ExecutionContext, Injectable, NestInterceptor } from '@nestjs/common';
import { Reflector } from '@nestjs/core';
import { Observable, of } from 'rxjs';

@Injectable()
export class ForbittenInterceptor implements NestInterceptor {

  static readonly FORBIT_KEY = Symbol('forbit');

  constructor(private readonly reflector: Reflector) {

  }

  intercept(context: ExecutionContext, next: CallHandler): Observable<any> {
    const clazz = context.getClass();
    const method = context.getHandler();
    console.log(`Executing ${clazz.name}.${method.name}`);
    const forbit = this.reflector.get(ForbittenInterceptor.FORBIT_KEY, clazz) ?? this.reflector.get(ForbittenInterceptor.FORBIT_KEY, method);
    if (forbit) {
      return of({ message: 'Forbidden' });
    }
    return next.handle();
  }
}
