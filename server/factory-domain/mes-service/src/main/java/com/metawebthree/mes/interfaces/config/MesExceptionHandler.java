package com.metawebthree.mes.interfaces.config;

import com.metawebthree.common.dto.ApiResponse;
import com.metawebthree.mes.domain.exception.ProcessRouteException;
import com.metawebthree.mes.domain.exception.EquipmentException;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.HttpStatus;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.bind.annotation.ResponseStatus;
import org.springframework.web.bind.annotation.RestControllerAdvice;

import jakarta.servlet.http.HttpServletRequest;

@Slf4j
@RestControllerAdvice
public class MesExceptionHandler {

    @ExceptionHandler(ProcessRouteException.class)
    @ResponseStatus(HttpStatus.BAD_REQUEST)
    public ApiResponse<?> handleProcessRouteException(ProcessRouteException e, HttpServletRequest request) {
        log.warn("工艺路线异常: {} - {} [{}]", 
            request.getRequestURI(), e.getMessage(), e.getCode());
        return ApiResponse.error(
            com.metawebthree.common.enums.ResponseStatus.PARAM_ERROR, 
            e.getMessage() + (e.getDetails() != null ? " - " + e.getDetails() : ""));
    }
    
    @ExceptionHandler(EquipmentException.class)
    @ResponseStatus(HttpStatus.BAD_REQUEST)
    public ApiResponse<?> handleEquipmentException(EquipmentException e, HttpServletRequest request) {
        log.warn("设备异常: {} - {} [{}]", 
            request.getRequestURI(), e.getMessage(), e.getCode());
        return ApiResponse.error(
            com.metawebthree.common.enums.ResponseStatus.PARAM_ERROR, 
            e.getMessage() + (e.getDetails() != null ? " - " + e.getDetails() : ""));
    }
    
    @ExceptionHandler(IllegalStateException.class)
    @ResponseStatus(HttpStatus.BAD_REQUEST)
    public ApiResponse<?> handleIllegalStateException(IllegalStateException e, HttpServletRequest request) {
        log.warn("状态异常: {} - {}", 
            request.getRequestURI(), e.getMessage());
        return ApiResponse.error(
            com.metawebthree.common.enums.ResponseStatus.PARAM_ERROR, 
            e.getMessage());
    }
}