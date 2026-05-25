package com.metawebthree.common.exception;

public enum ErrorCode {
    INVALID_PARAM("1001", "参数校验失败"),
    PARAM_MISSING("1002", "参数缺失"),
    PARAM_TYPE("1003", "参数类型错误"),
    METHOD_NOT_ALLOWED("1004", "不支持的请求方法"),
    NOT_FOUND("1005", "资源不存在"),
    FORBIDDEN("1007", "无权限访问"),
    INTERNAL_ERROR("9999", "系统错误");

    private final String code;
    private final String message;

    ErrorCode(String code, String message) {
        this.code = code;
        this.message = message;
    }

    public String getCode() {
        return code;
    }

    public String getMessage() {
        return message;
    }
}