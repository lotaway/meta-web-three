package com.metawebthree.recommendation.interfaces.controller;

import com.metawebthree.common.dto.ApiResponse;
import com.metawebthree.recommendation.application.aishopping.AiShoppingQueryService;
import com.metawebthree.recommendation.domain.aishopping.entity.AiProductMatch;
import com.metawebthree.recommendation.domain.aishopping.entity.TextCorrection;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;
import java.util.List;
import java.util.Map;
import org.springframework.http.MediaType;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

@Tag(name = "AI Shopping", description = "Customer-side AI shopping endpoints")
@RestController
@RequestMapping("/api/ai-shopping")
public class AiShoppingController {

    private final AiShoppingQueryService queryService;

    public AiShoppingController(AiShoppingQueryService queryService) {
        this.queryService = queryService;
    }

    @Operation(summary = "Correct text using LLM or local dictionary")
    @PostMapping("/text-correct")
    public ApiResponse<TextCorrection> textCorrect(@RequestBody Map<String, String> request) {
        return ApiResponse.success(queryService.correctText(request.getOrDefault("text", "")));
    }

    @Operation(summary = "Smart match: semantic search over product text vectors")
    @PostMapping("/smart-match")
    public ApiResponse<List<AiProductMatch>> smartMatch(@RequestBody Map<String, Object> request) {
        String query = (String) request.getOrDefault("query", "");
        Integer topK = request.get("topK") != null ? ((Number) request.get("topK")).intValue() : null;
        return ApiResponse.success(queryService.smartMatch(query, topK));
    }

    @Operation(summary = "Image search: semantic search over product image vectors")
    @PostMapping(value = "/image-search", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
    public ApiResponse<List<AiProductMatch>> imageSearch(
            @RequestParam("image") MultipartFile image,
            @RequestParam(value = "topK", required = false) Integer topK) throws Exception {
        return ApiResponse.success(queryService.imageSearch(image.getBytes(), topK));
    }

    @Operation(summary = "One-stop search: text correction then smart match")
    @PostMapping("/search")
    public ApiResponse<Map<String, Object>> search(@RequestBody Map<String, Object> request) {
        String q = (String) request.getOrDefault("q", request.getOrDefault("query", ""));
        Integer topK = request.get("topK") != null ? ((Number) request.get("topK")).intValue() : null;
        Long userId = request.get("userId") != null ? ((Number) request.get("userId")).longValue() : null;
        return ApiResponse.success(queryService.combinedSearch(q, topK, userId));
    }
}
