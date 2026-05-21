package com.metawebthree.action.interfaces.web;

import com.metawebthree.action.application.UserActionService;
import com.metawebthree.action.domain.model.BrandAttention;
import com.metawebthree.action.domain.model.ProductCollection;
import com.metawebthree.action.domain.model.ReadHistory;
import com.metawebthree.common.constants.HeaderConstants;
import com.metawebthree.common.dto.ApiResponse;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;
import lombok.Getter;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping("/member")
@RequiredArgsConstructor
@Tag(name = "Member Action Controller", description = "用户行为接口 (收藏、足迹、关注) - 兼容参考项目路径")
public class MemberActionController {

    private final UserActionService userActionService;

    @Operation(summary = "添加品牌关注")
    @PostMapping("/attention/add")
    public ApiResponse<Void> addAttention(@RequestHeader(HeaderConstants.USER_ID) Long userId,
                                           @RequestBody AttentionParam param) {
        userActionService.addAttention(userId, param.getBrandId(), param.getBrandName(), param.getBrandLogo());
        return ApiResponse.success();
    }

    @Operation(summary = "获取品牌关注列表")
    @GetMapping("/attention/list")
    public ApiResponse<List<BrandAttention>> listAttention(@RequestHeader(HeaderConstants.USER_ID) Long userId) {
        return ApiResponse.success(userActionService.listAttentions(userId));
    }

    @Operation(summary = "取消品牌关注")
    @DeleteMapping("/attention")
    public ApiResponse<Void> deleteAttention(@RequestHeader(HeaderConstants.USER_ID) Long userId,
                                              @RequestParam Long brandId) {
        userActionService.deleteAttention(userId, brandId);
        return ApiResponse.success();
    }

    @Operation(summary = "获取品牌关注详情")
    @GetMapping("/attention/detail")
    public ApiResponse<BrandAttention> detailAttention(@RequestHeader(HeaderConstants.USER_ID) Long userId,
                                                        @RequestParam Long brandId) {
        return ApiResponse.success(userActionService.getAttentionDetail(userId, brandId));
    }

    @Operation(summary = "清空品牌关注")
    @DeleteMapping("/attention/clear")
    public ApiResponse<Void> clearAttention(@RequestHeader(HeaderConstants.USER_ID) Long userId) {
        userActionService.clearAttentions(userId);
        return ApiResponse.success();
    }

    @Operation(summary = "添加商品收藏")
    @PostMapping("/productCollection/add")
    public ApiResponse<Void> addCollection(@RequestHeader(HeaderConstants.USER_ID) Long userId,
                                            @RequestBody CollectionParam param) {
        userActionService.addCollection(userId, param.getProductId(), param.getProductName(), param.getProductPic());
        return ApiResponse.success();
    }

    @Operation(summary = "获取商品收藏列表")
    @GetMapping("/productCollection/list")
    public ApiResponse<List<ProductCollection>> listCollection(@RequestHeader(HeaderConstants.USER_ID) Long userId) {
        return ApiResponse.success(userActionService.listCollections(userId));
    }

    @Operation(summary = "取消商品收藏")
    @DeleteMapping("/productCollection")
    public ApiResponse<Void> deleteCollection(@RequestHeader(HeaderConstants.USER_ID) Long userId,
                                               @RequestParam Long productId) {
        userActionService.deleteCollection(userId, productId);
        return ApiResponse.success();
    }

    @Operation(summary = "获取商品收藏详情")
    @GetMapping("/productCollection/detail")
    public ApiResponse<ProductCollection> detailCollection(@RequestHeader(HeaderConstants.USER_ID) Long userId,
                                                            @RequestParam Long productId) {
        return ApiResponse.success(userActionService.getCollectionDetail(userId, productId));
    }

    @Operation(summary = "清空商品收藏")
    @DeleteMapping("/productCollection/clear")
    public ApiResponse<Void> clearCollection(@RequestHeader(HeaderConstants.USER_ID) Long userId) {
        userActionService.clearCollections(userId);
        return ApiResponse.success();
    }

    @Operation(summary = "添加浏览记录")
    @PostMapping("/readHistory/create")
    public ApiResponse<Void> addHistory(@RequestHeader(HeaderConstants.USER_ID) Long userId,
                                         @RequestBody HistoryParam param) {
        userActionService.addHistory(userId, param.getProductId(), param.getProductName(), param.getProductPic());
        return ApiResponse.success();
    }

    @Operation(summary = "获取浏览记录列表")
    @GetMapping("/readHistory/list")
    public ApiResponse<List<ReadHistory>> listHistory(@RequestHeader(HeaderConstants.USER_ID) Long userId) {
        return ApiResponse.success(userActionService.listHistory(userId));
    }

    @Operation(summary = "删除浏览记录")
    @DeleteMapping("/readHistory")
    public ApiResponse<Void> deleteHistory(@RequestHeader(HeaderConstants.USER_ID) Long userId,
                                            @RequestParam Long productId) {
        userActionService.deleteHistory(userId, productId);
        return ApiResponse.success();
    }

    @Operation(summary = "清空浏览记录")
    @DeleteMapping("/readHistory/clear")
    public ApiResponse<Void> clearHistory(@RequestHeader(HeaderConstants.USER_ID) Long userId) {
        userActionService.clearHistory(userId);
        return ApiResponse.success();
    }

    @Getter
    public static class AttentionParam {
        private Long brandId;
        private String brandName;
        private String brandLogo;
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
}
