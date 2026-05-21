package com.metawebthree.cs.ai.tools;

import com.metawebthree.cs.dto.AiToolRequest;
import com.metawebthree.cs.dto.AiToolResult;

import java.util.Map;

public abstract class AiTool {
    private final String name;
    private final String description;

    protected AiTool(String name, String description) {
        this.name = name;
        this.description = description;
    }

    public abstract AiToolResult execute(AiToolRequest request);

    public abstract Map<String, Object> getParameterSchema();

    public String getName() { return name; }
    public String getDescription() { return description; }
}
