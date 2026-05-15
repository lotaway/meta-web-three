package com.metawebthree.cs.dto;

import lombok.Data;
import java.util.Map;

@Data
public class AiToolRequest {
    private String tool;
    private String sessionId;
    private Long userId;
    private Map<String, Object> params;
}
