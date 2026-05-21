package com.metawebthree.cs.interfaces.web;

import com.metawebthree.common.dto.ApiResponse;
import com.metawebthree.cs.dto.AiToolRequest;
import com.metawebthree.cs.dto.AiToolResponse;
import com.metawebthree.cs.ai.tools.AiToolRegistry;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.util.Map;

@RestController
@RequestMapping("/cs/ai/tools")
@Tag(name = "AI Tool Controller", description = "AI 工具调用接口")
public class AiToolController {
    private final AiToolRegistry toolRegistry;

    public AiToolController(AiToolRegistry toolRegistry) {
        this.toolRegistry = toolRegistry;
    }

    @Operation(summary = "调用 AI 工具")
    @PostMapping("/execute")
    public ApiResponse<AiToolResponse> execute(@RequestBody AiToolRequest request) {
        return toolRegistry.getTool(request.getTool())
                .map(tool -> {
                    var result = tool.execute(request);
                    if (result.isSuccess()) {
                        return ApiResponse.success(AiToolResponse.ok(result.getResult()));
                    }
                    return ApiResponse.success(AiToolResponse.fail(
                            result.getResult().toString(), result.getTool()));
                })
                .orElse(ApiResponse.success(
                        AiToolResponse.fail("未知工具: " + request.getTool(), "TOOL_NOT_FOUND")));
    }

    @Operation(summary = "获取工具列表及参数描述")
    @GetMapping("/schemas")
    public ApiResponse<Map<String, Object>> getSchemas() {
        return ApiResponse.success(toolRegistry.getToolSchemas());
    }
}
