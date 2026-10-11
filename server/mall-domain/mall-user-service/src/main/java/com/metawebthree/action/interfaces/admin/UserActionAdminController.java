package com.metawebthree.action.interfaces.admin;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.metawebthree.common.dto.ApiResponse;
import com.metawebthree.common.enums.ResponseStatus;
import com.metawebthree.action.domain.model.BrandAttention;
import com.metawebthree.action.domain.model.ProductCollection;
import com.metawebthree.action.domain.model.ProductComment;
import com.metawebthree.action.domain.model.ReadHistory;
import com.metawebthree.action.infrastructure.persistence.mapper.BrandAttentionMapper;
import com.metawebthree.action.infrastructure.persistence.mapper.ProductCollectionMapper;
import com.metawebthree.action.infrastructure.persistence.mapper.ProductCommentMapper;
import com.metawebthree.action.infrastructure.persistence.mapper.ReadHistoryMapper;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

@RestController
@RequestMapping("/api/admin/user-action")
public class UserActionAdminController {

    @Autowired
    private ProductCollectionMapper collectionMapper;

    @Autowired
    private ReadHistoryMapper historyMapper;

    @Autowired
    private BrandAttentionMapper attentionMapper;

    @Autowired
    private ProductCommentMapper commentMapper;

    @GetMapping("/collection/list")
    public ApiResponse<Map<String, Object>> listCollections(
            @RequestParam(defaultValue = "1") Integer pageNum,
            @RequestParam(defaultValue = "10") Integer pageSize,
            @RequestParam(required = false) Long userId,
            @RequestParam(required = false) Long productId,
            @RequestParam(required = false) String productName) {

        LambdaQueryWrapper<ProductCollection> wrapper = new LambdaQueryWrapper<ProductCollection>()
                .eq(userId != null, ProductCollection::getUserId, userId)
                .eq(productId != null, ProductCollection::getProductId, productId)
                .like(productName != null && !productName.isEmpty(), ProductCollection::getProductName, productName)
                .orderByDesc(ProductCollection::getCreateTime);

        Page<ProductCollection> page = new Page<>(pageNum, pageSize);
        Page<ProductCollection> result = collectionMapper.selectPage(page, wrapper);

        Map<String, Object> data = new HashMap<>();
        data.put("list", result.getRecords());
        data.put("total", result.getTotal());
        data.put("pageNum", pageNum);
        data.put("pageSize", pageSize);

        return ApiResponse.success(data);
    }

    @GetMapping("/collection/{id}")
    public ApiResponse<ProductCollection> getCollectionById(@PathVariable Long id) {
        ProductCollection collection = collectionMapper.selectById(id);
        if (collection == null) {
            return ApiResponse.error(ResponseStatus.NOT_FOUND, "Collection not found");
        }
        return ApiResponse.success(collection);
    }

    @DeleteMapping("/collection/{id}")
    public ApiResponse<Void> deleteCollection(@PathVariable Long id) {
        collectionMapper.deleteById(id);
        return ApiResponse.success();
    }

    @DeleteMapping("/collection/batch")
    public ApiResponse<Void> batchDeleteCollections(@RequestBody List<Long> ids) {
        collectionMapper.deleteBatchIds(ids);
        return ApiResponse.success();
    }

    @GetMapping("/history/list")
    public ApiResponse<Map<String, Object>> listHistories(
            @RequestParam(defaultValue = "1") Integer pageNum,
            @RequestParam(defaultValue = "10") Integer pageSize,
            @RequestParam(required = false) Long userId,
            @RequestParam(required = false) Long productId,
            @RequestParam(required = false) String productName) {

        LambdaQueryWrapper<ReadHistory> wrapper = new LambdaQueryWrapper<ReadHistory>()
                .eq(userId != null, ReadHistory::getUserId, userId)
                .eq(productId != null, ReadHistory::getProductId, productId)
                .like(productName != null && !productName.isEmpty(), ReadHistory::getProductName, productName)
                .orderByDesc(ReadHistory::getCreateTime);

        Page<ReadHistory> page = new Page<>(pageNum, pageSize);
        Page<ReadHistory> result = historyMapper.selectPage(page, wrapper);

        Map<String, Object> data = new HashMap<>();
        data.put("list", result.getRecords());
        data.put("total", result.getTotal());
        data.put("pageNum", pageNum);
        data.put("pageSize", pageSize);

        return ApiResponse.success(data);
    }

    @DeleteMapping("/history/{id}")
    public ApiResponse<Void> deleteHistory(@PathVariable Long id) {
        historyMapper.deleteById(id);
        return ApiResponse.success();
    }

    @DeleteMapping("/history/batch")
    public ApiResponse<Void> batchDeleteHistories(@RequestBody List<Long> ids) {
        historyMapper.deleteBatchIds(ids);
        return ApiResponse.success();
    }

    @GetMapping("/attention/list")
    public ApiResponse<Map<String, Object>> listAttentions(
            @RequestParam(defaultValue = "1") Integer pageNum,
            @RequestParam(defaultValue = "10") Integer pageSize,
            @RequestParam(required = false) Long userId,
            @RequestParam(required = false) Long brandId,
            @RequestParam(required = false) String brandName) {

        LambdaQueryWrapper<BrandAttention> wrapper = new LambdaQueryWrapper<BrandAttention>()
                .eq(userId != null, BrandAttention::getUserId, userId)
                .eq(brandId != null, BrandAttention::getBrandId, brandId)
                .like(brandName != null && !brandName.isEmpty(), BrandAttention::getBrandName, brandName)
                .orderByDesc(BrandAttention::getCreateTime);

        Page<BrandAttention> page = new Page<>(pageNum, pageSize);
        Page<BrandAttention> result = attentionMapper.selectPage(page, wrapper);

        Map<String, Object> data = new HashMap<>();
        data.put("list", result.getRecords());
        data.put("total", result.getTotal());
        data.put("pageNum", pageNum);
        data.put("pageSize", pageSize);

        return ApiResponse.success(data);
    }

    @DeleteMapping("/attention/{id}")
    public ApiResponse<Void> deleteAttention(@PathVariable Long id) {
        attentionMapper.deleteById(id);
        return ApiResponse.success();
    }

    @DeleteMapping("/attention/batch")
    public ApiResponse<Void> batchDeleteAttentions(@RequestBody List<Long> ids) {
        attentionMapper.deleteBatchIds(ids);
        return ApiResponse.success();
    }

    @GetMapping("/comment/list")
    public ApiResponse<Map<String, Object>> listComments(
            @RequestParam(defaultValue = "1") Integer pageNum,
            @RequestParam(defaultValue = "10") Integer pageSize,
            @RequestParam(required = false) Long userId,
            @RequestParam(required = false) Long productId,
            @RequestParam(required = false) String productName,
            @RequestParam(required = false) Integer showStatus,
            @RequestParam(required = false) Integer star) {

        LambdaQueryWrapper<ProductComment> wrapper = new LambdaQueryWrapper<ProductComment>()
                .eq(userId != null, ProductComment::getUserId, userId)
                .eq(productId != null, ProductComment::getProductId, productId)
                .like(productName != null && !productName.isEmpty(), ProductComment::getProductName, productName)
                .eq(showStatus != null, ProductComment::getShowStatus, showStatus)
                .eq(star != null, ProductComment::getStar, star)
                .orderByDesc(ProductComment::getCreateTime);

        Page<ProductComment> page = new Page<>(pageNum, pageSize);
        Page<ProductComment> result = commentMapper.selectPage(page, wrapper);

        Map<String, Object> data = new HashMap<>();
        data.put("list", result.getRecords());
        data.put("total", result.getTotal());
        data.put("pageNum", pageNum);
        data.put("pageSize", pageSize);

        return ApiResponse.success(data);
    }

    @GetMapping("/comment/{id}")
    public ApiResponse<ProductComment> getCommentById(@PathVariable Long id) {
        ProductComment comment = commentMapper.selectById(id);
        if (comment == null) {
            return ApiResponse.error(ResponseStatus.NOT_FOUND, "Comment not found");
        }
        return ApiResponse.success(comment);
    }

    @PutMapping("/comment/{id}/status")
    public ApiResponse<Void> updateCommentStatus(@PathVariable Long id, @RequestParam Integer showStatus) {
        ProductComment comment = commentMapper.selectById(id);
        if (comment == null) {
            return ApiResponse.error(ResponseStatus.NOT_FOUND, "Comment not found");
        }
        comment.setShowStatus(showStatus);
        commentMapper.updateById(comment);
        return ApiResponse.success();
    }

    @DeleteMapping("/comment/{id}")
    public ApiResponse<Void> deleteComment(@PathVariable Long id) {
        commentMapper.deleteById(id);
        return ApiResponse.success();
    }

    @DeleteMapping("/comment/batch")
    public ApiResponse<Void> batchDeleteComments(@RequestBody List<Long> ids) {
        commentMapper.deleteBatchIds(ids);
        return ApiResponse.success();
    }

    @GetMapping("/statistics")
    public ApiResponse<Map<String, Object>> getStatistics() {
        long totalCollections = collectionMapper.selectCount(null);
        long totalHistories = historyMapper.selectCount(null);
        long totalAttentions = attentionMapper.selectCount(null);
        long totalComments = commentMapper.selectCount(null);
        long visibleComments = commentMapper.selectCount(new LambdaQueryWrapper<ProductComment>().eq(ProductComment::getShowStatus, 1));

        Map<String, Object> stats = new HashMap<>();
        stats.put("totalCollections", totalCollections);
        stats.put("totalHistories", totalHistories);
        stats.put("totalAttentions", totalAttentions);
        stats.put("totalComments", totalComments);
        stats.put("visibleComments", visibleComments);

        return ApiResponse.success(stats);
    }
}