import { CanActivate, ExecutionContext, Injectable } from '@nestjs/common';
import { Reflector } from '@nestjs/core';
import { Observable } from 'rxjs';

@Injectable()
export class AuthrozationGuard implements CanActivate {

  constructor(private readonly reflector: Reflector) {

  }

  canActivate(
    context: ExecutionContext,
  ): boolean | Promise<boolean> | Observable<boolean> {
    // const roles = this.reflector.get<string[]>('roles', context.getHandler());
    const request = context.switchToHttp().getRequest<Request>();
    if (request.url.endsWith('/login'))
      return true;
    const [type, token] = request.headers.get("Authorization")?.split(" ");
    if (!type || type !== "Bearer" || !token)
      return false;
    return true;
  }
}
