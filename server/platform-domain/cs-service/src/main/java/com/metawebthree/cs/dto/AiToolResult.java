package com.metawebthree.cs.dto;

import lombok.Data;
import java.util.Map;

@Data
public class AiToolResult {
    private String tool;
    private boolean success;
    private Object result;

    public static AiToolResult success(String tool, Object result) {
        AiToolResult r = new AiToolResult();
        r.tool = tool;
        r.success = true;
        r.result = result;
        return r;
    }

    public static AiToolResult failure(String tool, String error) {
        AiToolResult r = new AiToolResult();
        r.tool = tool;
        r.success = false;
        r.result = Map.of("error", error);
        return r;
    }
}
