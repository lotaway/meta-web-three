package com.metawebthree.common.dto;


import lombok.Data;

import java.util.Objects;

import com.metawebthree.common.enums.ResponseStatus;

@Data
public class ApiResponse<D> extends IBaseResponse<D> {

    public ApiResponse() {

    }

    public ApiResponse(ResponseStatus status, String message) {
        this.status = status;
        this.message = message;
    }

    public ApiResponse(ResponseStatus status, String message, D data) {
        this.status = status;
        this.message = message;
        this.data = data;
    }

    public ResponseStatus getStatus() {
        return status;
    }

    public String getMessage() {
        return message;
    }

    public Object getData() {
        return data;
    }

    public static <D> ApiResponse<D> success() {
        return new ApiResponse<>(ResponseStatus.SUCCESS, "success");
    }

    public static <D> ApiResponse<D> success(D data) {
        return new ApiResponse<>(ResponseStatus.SUCCESS, "success", data);
    }

    public static ApiResponse<Exception> error() {
        return new ApiResponse<>(ResponseStatus.ERROR, "error");
    }

    public static ApiResponse<Exception> error(String errMessage) {
        return new ApiResponse<>(ResponseStatus.ERROR, errMessage);
    }

    public static <T> ApiResponse<T> error(String errMessage, Class<T> clazz) {
        return new ApiResponse<>(ResponseStatus.ERROR, errMessage);
    }

    public static ApiResponse<Exception> error(Exception e) {
        return new ApiResponse<>(ResponseStatus.ERROR, "error", e);
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        if (!(o instanceof ApiResponse)) {
            return false;
        }
        ApiResponse<?> that = (ApiResponse<?>) o;
        return status == that.status && message.equals(that.message) && data.equals(that.data);
    }

    @Override
    public int hashCode() {
        return Objects.hash(status, message, data);
    }

    @Override
    public String toString() {
        return "{'status':" + status + ",'message':'" + message + ", 'data':" + data + '}';
    }
}
