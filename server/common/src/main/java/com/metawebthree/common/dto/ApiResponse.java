package com.metawebthree.common.dto;

import com.metawebthree.common.enums.ResponseStatus;
import lombok.Data;

import java.util.Objects;

@Data
public class ApiResponse<D> {
    private String code;
    private String message;
    private D data;
    private Long timestamp;

    public ApiResponse() {
        this.timestamp = System.currentTimeMillis();
    }

    public ApiResponse(ResponseStatus status, D data) {
        this.code = status.getCode();
        this.message = status.getMessage();
        this.data = data;
        this.timestamp = System.currentTimeMillis();
    }

    public ApiResponse(String code, String message, D data) {
        this.code = code;
        this.message = message;
        this.data = data;
        this.timestamp = System.currentTimeMillis();
    }

    public static <D> ApiResponse<D> success() {
        return new ApiResponse<>(ResponseStatus.SUCCESS, null);
    }

    public static <D> ApiResponse<D> success(D data) {
        return new ApiResponse<>(ResponseStatus.SUCCESS, data);
    }

    public static <D> ApiResponse<D> error(ResponseStatus status) {
        return new ApiResponse<>(status, null);
    }

    public static <D> ApiResponse<D> error(ResponseStatus status, String customMessage) {
        ApiResponse<D> response = new ApiResponse<>(status, null);
        response.setMessage(customMessage);
        return response;
    }
}
