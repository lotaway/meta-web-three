package com.metawebthree.common.exception;

import com.metawebthree.common.enums.ResponseStatus;
import lombok.Getter;

@Getter
public class BusinessException extends RuntimeException {
    private final ResponseStatus status;
    private final String message;

    public BusinessException(ResponseStatus status) {
        super(status.getMessage());
        this.status = status;
        this.message = status.getMessage();
    }

    public BusinessException(ResponseStatus status, String message) {
        super(message);
        this.status = status;
        this.message = message;
    }

    public BusinessException(ResponseStatus status, String message, Throwable cause) {
        super(message, cause);
        this.status = status;
        this.message = message;
    }
}