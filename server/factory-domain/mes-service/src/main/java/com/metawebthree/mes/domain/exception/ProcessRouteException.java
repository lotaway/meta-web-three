package com.metawebthree.mes.domain.exception;

public class ProcessRouteException extends RuntimeException {
    
    private final String code;
    private final String details;
    
    public ProcessRouteException(String code, String message) {
        super(message);
        this.code = code;
        this.details = null;
    }
    
    public ProcessRouteException(String code, String message, String details) {
        super(message);
        this.code = code;
        this.details = details;
    }
    
    public ProcessRouteException(String code, String message, Throwable cause) {
        super(message, cause);
        this.code = code;
        this.details = null;
    }
    
    public String getCode() {
        return code;
    }
    
    public String getDetails() {
        return details;
    }
    
    public static ProcessRouteException notFound(Long id) {
        return new ProcessRouteException("PROCESS_ROUTE_NOT_FOUND", 
            "工艺路线不存在", "ID: " + id);
    }
    
    public static ProcessRouteException notFound(String routeCode) {
        return new ProcessRouteException("PROCESS_ROUTE_NOT_FOUND", 
            "工艺路线不存在", "编码: " + routeCode);
    }
    
    public static ProcessRouteException validationFailed(String errors) {
        return new ProcessRouteException("PROCESS_ROUTE_VALIDATION_FAILED", 
            "工艺路线验证失败", errors);
    }
    
    public static ProcessRouteException activationFailed(String reason) {
        return new ProcessRouteException("PROCESS_ROUTE_ACTIVATION_FAILED", 
            "工艺路线激活失败", reason);
    }
}