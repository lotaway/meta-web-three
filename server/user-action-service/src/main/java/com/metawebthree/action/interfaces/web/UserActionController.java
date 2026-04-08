package com.metawebthree.action.interfaces.web;

import com.metawebthree.action.application.UserActionService;
import com.metawebthree.common.constants.HeaderConstants;
import com.metawebthree.common.dto.ApiResponse;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;
import lombok.Getter;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping("/v1/action")
@RequiredArgsConstructor
@Tag(name = "User Action Controller", description = "用户行为接口 (收藏、足迹、关注)")
public class UserActionController {

    private final UserActionService userActionService;

    @Operation(summary = "添加商品收藏")
    @PostMapping("/productCollection/add")
    public ApiResponse<Void> addCollection(@RequestHeader(HeaderConstants.USER_ID) Long userId, @RequestBody CollectionParam param) {
        userActionService.addCollection(userId, param.getProductId(), param.getProductName(), param.getProductPic());
        return ApiResponse.success();
    }

    @Operation(summary = "获取收藏列表")
    @GetMapping("/productCollection/list")
    public ApiResponse<List<?>> listCollection(@RequestHeader(HeaderConstants.USER_ID) Long userId) {
        return ApiResponse.success(userActionService.listCollections(userId));
    }

    @Operation(summary = "取消商品收藏")
    @DeleteMapping("/productCollection/delete")
    public ApiResponse<Void> deleteCollection(@RequestHeader(HeaderConstants.USER_ID) Long userId, @RequestParam Long productId) {
        userActionService.deleteCollection(userId, productId);
        return ApiResponse.success();
    }

    @Operation(summary = "添加浏览记录")
    @PostMapping("/readHistory/create")
    public ApiResponse<Void> addHistory(@RequestHeader(HeaderConstants.USER_ID) Long userId, @RequestBody HistoryParam param) {
        userActionService.addHistory(userId, param.getProductId(), param.getProductName(), param.getProductPic());
        return ApiResponse.success();
    }

    @Operation(summary = "获取浏览记录列表")
    @GetMapping("/readHistory/list")
    public ApiResponse<List<?>> listHistory(@RequestHeader(HeaderConstants.USER_ID) Long userId) {
        return ApiResponse.success(userActionService.listHistory(userId));
    }

    @Operation(summary = "添加品牌关注")
    @PostMapping("/brandAttention/add")
    public ApiResponse<Void> addAttention(@RequestHeader(HeaderConstants.USER_ID) Long userId, @RequestBody AttentionParam param) {
        userActionService.addAttention(userId, param.getBrandId(), param.getBrandName(), param.getBrandLogo());
        return ApiResponse.success();
    }

    @Operation(summary = "获取关注列表")
    @GetMapping("/brandAttention/list")
    public ApiResponse<List<?>> listAttention(@RequestHeader(HeaderConstants.USER_ID) Long userId) {
        return ApiResponse.success(userActionService.listAttentions(userId));
    }

    @Getter
    public static class CollectionParam {
        private Long productId;
        private String productName;
        private String productPic;
    }

    @Getter
    public static class HistoryParam {
        private Long productId;
        private String productName;
        private String productPic;
    }

    @Getter
    public static class AttentionParam {
        private Long brandId;
        private String brandName;
        private String brandLogo;
    }
}
