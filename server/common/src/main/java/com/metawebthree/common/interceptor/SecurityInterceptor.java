package com.metawebthree.common.interceptor;

import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;
import org.springframework.web.servlet.HandlerInterceptor;
import org.springframework.web.servlet.ModelAndView;

import com.metawebthree.common.enums.ResponseStatus;

@Slf4j
@Component
public class SecurityInterceptor implements HandlerInterceptor {

    public boolean preHandle(HttpServletRequest request, HttpServletResponse response, Object handler) throws Exception {
        if (request.getRequestURL().indexOf("/config") > -1) {
            response.setContentType("application/json;charset=UTF-8");
            response.getWriter().write("{\"code\":\"" + ResponseStatus.SYSTEM_ERROR.getCode() + "\",\"message\":\"security interceptor no pass\"}");
            return false;
        }
        return true;
    }

    public void postHandle(HttpServletRequest request, HttpServletResponse response, Object handler, ModelAndView modelAndView) throws Exception {

    }

    public void afterCompletion(HttpServletRequest request, HttpServletResponse response, Object handler, Exception ex) throws Exception {

    }

}
