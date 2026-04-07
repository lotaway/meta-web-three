package com.metawebthree.common.dto;

import com.metawebthree.common.enums.ResponseStatus;
import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;

import java.util.Objects;

@Data
@Schema(description = "通用 API 响应")
public class ApiResponse<D> {
    @Schema(description = "响应码")
    private String code;
    @Schema(description = "响应消息")
    private String message;
    @Schema(description = "响应数据")
    private D data;
    @Schema(description = "时间戳")
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
