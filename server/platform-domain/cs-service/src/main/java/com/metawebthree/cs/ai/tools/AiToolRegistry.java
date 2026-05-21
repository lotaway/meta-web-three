package com.metawebthree.cs.ai.tools;

import java.util.Collection;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Optional;

public class AiToolRegistry {
    private final Map<String, AiTool> tools = new LinkedHashMap<>();

    public void register(AiTool tool) {
        tools.put(tool.getName(), tool);
    }

    public Optional<AiTool> getTool(String name) {
        return Optional.ofNullable(tools.get(name));
    }

    public Collection<AiTool> getAllTools() {
        return tools.values();
    }

    public Map<String, Object> getToolSchemas() {
        Map<String, Object> schemas = new LinkedHashMap<>();
        for (AiTool tool : tools.values()) {
            schemas.put(tool.getName(), Map.of(
                    "description", tool.getDescription(),
                    "parameters", tool.getParameterSchema()
            ));
        }
        return schemas;
    }
}
