package com.metawebthree.cs.dto;

import lombok.Data;

@Data
public class AiToolResponse {
    private boolean success;
    private String message;
    private Object data;
    private String errorCode;

    public static AiToolResponse ok(Object data) {
        AiToolResponse r = new AiToolResponse();
        r.success = true;
        r.message = "ok";
        r.data = data;
        return r;
    }

    public static AiToolResponse fail(String message, String errorCode) {
        AiToolResponse r = new AiToolResponse();
        r.success = false;
        r.message = message;
        r.errorCode = errorCode;
        return r;
    }
}
