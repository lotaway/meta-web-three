package com.metawebthree.action.interfaces.web;

import com.metawebthree.action.application.UserActionService;
import com.metawebthree.action.domain.model.ProductComment;
import com.metawebthree.common.constants.HeaderConstants;
import com.metawebthree.common.dto.ApiResponse;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;
import lombok.Getter;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping("/action")
@RequiredArgsConstructor
@Tag(name = "User Action Controller", description = "用户行为接口 (收藏、足迹、关注、评论)")
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

    @Operation(summary = "发表评论")
    @PostMapping("/comment/create")
    public ApiResponse<Void> addComment(@RequestHeader(HeaderConstants.USER_ID) Long userId,
            @RequestParam(required = false) String nickName,
            @RequestBody CommentParam param) {
        userActionService.addComment(userId, nickName, param.getProductId(), param.getProductName(),
                param.getStar(), param.getContent(), param.getPics(), param.getProductAttribute());
        return ApiResponse.success();
    }

    @Operation(summary = "商品评论列表")
    @GetMapping("/comment/listByProduct")
    public ApiResponse<List<ProductComment>> listCommentsByProduct(@RequestParam Long productId) {
        return ApiResponse.success(userActionService.listCommentsByProduct(productId));
    }

    @Operation(summary = "评论详情")
    @GetMapping("/comment/detail")
    public ApiResponse<ProductComment> getCommentDetail(@RequestParam Long commentId) {
        return ApiResponse.success(userActionService.getCommentDetail(commentId));
    }

    @Operation(summary = "点赞评论")
    @PostMapping("/comment/like")
    public ApiResponse<Void> likeComment(@RequestParam Long commentId) {
        userActionService.likeComment(commentId);
        return ApiResponse.success();
    }

    @Operation(summary = "我的评论列表")
    @GetMapping("/comment/listByUser")
    public ApiResponse<List<ProductComment>> listCommentsByUser(@RequestHeader(HeaderConstants.USER_ID) Long userId) {
        return ApiResponse.success(userActionService.listCommentsByUser(userId));
    }

    @Getter
    public static class CommentParam {
        private Long productId;
        private String productName;
        private Integer star;
        private String content;
        private String pics;
        private String productAttribute;
    }
}
