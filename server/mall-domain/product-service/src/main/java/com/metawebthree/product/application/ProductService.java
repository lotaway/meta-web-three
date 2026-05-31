package com.metawebthree.product.application;

import com.baomidou.mybatisplus.core.conditions.query.QueryWrapper;
import com.baomidou.mybatisplus.core.toolkit.IdWorker;
import com.metawebthree.product.dto.ProductDTO;
import com.metawebthree.product.dto.ProductDetailDTO;
import com.metawebthree.common.utils.RocketMQ.MQProducer;
import com.metawebthree.image.ProductImageService;
import com.metawebthree.product.application.event.ProductEventType;
import com.metawebthree.product.domain.exception.ProductDomainException;
import com.metawebthree.product.domain.exception.ProductErrorCode;
import com.metawebthree.product.domain.model.*;
import com.metawebthree.product.infrastructure.persistence.mapper.*;
import java.math.BigDecimal;
import java.util.*;
import java.util.function.Function;
import java.util.stream.Collectors;

import lombok.extern.slf4j.Slf4j;

import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;

import io.github.resilience4j.circuitbreaker.annotation.CircuitBreaker;
import io.github.resilience4j.ratelimiter.annotation.RateLimiter;

@Service("productAppService")
@Slf4j
public class ProductService {
    private static final String PRODUCT_EVENT_TOPIC = "product-events";

    private final MQProducer mqProducer;
    private final ProductImageService productImageService;
    private final ProductMapper productMapper;
    private final ProductEntityMapper productEntityMapper;
    private final ProductStatsMapper productStatsMapper;
    private final ProductLimitsMapper productLimitsMapper;

    public ProductService(MQProducer mqProducer, ProductImageService productImageService,
            ProductMapper productMapper, ProductEntityMapper productEntityMapper,
            ProductStatsMapper productStatsMapper, ProductLimitsMapper productLimitsMapper) {
        this.mqProducer = mqProducer;
        this.productImageService = productImageService;
        this.productMapper = productMapper;
        this.productEntityMapper = productEntityMapper;
        this.productStatsMapper = productStatsMapper;
        this.productLimitsMapper = productLimitsMapper;
    }

    @RateLimiter(name = "getProductDetail")
    @CircuitBreaker(name = "productService", fallbackMethod = "getProductDetailFallback")
    public ProductDetailDTO getProductDetail(Integer id) {
        ProductDO product = productMapper.selectById(id);
        if (product == null) {
            return null;
        }

        List<ProductEntityDO> entities = productEntityMapper.selectList(
                new QueryWrapper<ProductEntityDO>().eq("product_id", id));

        ProductEntityDO defaultEntity = findDefaultEntity(entities);

        ProductStatsDO stats = productStatsMapper.selectById(id);
        ProductLimitsDO limits = productLimitsMapper.selectById(id);

        return buildProductDetail(product, defaultEntity, stats, limits);
    }

    private ProductDetailDTO getProductDetailFallback(Integer id, Throwable t) {
        log.error("商品详情查询失败，触发限流或熔断: productId={}, error={}", id, t.getMessage());
        throw new RuntimeException("系统繁忙，请稍后再试");
    }

    private ProductEntityDO findDefaultEntity(List<ProductEntityDO> entities) {
        return entities.stream()
                .min(Comparator.comparing(ProductEntityDO::getSalePrice))
                .orElseGet(ProductEntityDO::new);
    }

    private ProductDetailDTO buildProductDetail(ProductDO product, ProductEntityDO defaultEntity,
            ProductStatsDO stats, ProductLimitsDO limits) {
        ProductDetailDTO detail = new ProductDetailDTO();
        mapBasicInfo(detail, product);
        mapStatsInfo(detail, stats);
        mapLimitsInfo(detail, limits);
        mapEntityInfo(detail, defaultEntity);
        initCollections(detail);
        return detail;
    }

    private void mapBasicInfo(ProductDetailDTO detail, ProductDO product) {
        detail.setId(product.getId());
        detail.setGoodsName(product.getProductName());
        detail.setGoodsNo(product.getProductNo());
        detail.setGoodsRemark(product.getProductRemark());
    }

    private void mapStatsInfo(ProductDetailDTO detail, ProductStatsDO stats) {
        if (stats == null)
            return;
        detail.setCommentNumber(stats.getCommentNumber());
        detail.setScoreNumber(stats.getScoreNumber());
        detail.setScores(convertScore(stats.getScores()));
    }

    private void mapLimitsInfo(ProductDetailDTO detail, ProductLimitsDO limits) {
        if (limits == null)
            return;
        detail.setPurchase(limits.getPurchase());
    }

    private void mapEntityInfo(ProductDetailDTO detail, ProductEntityDO entity) {
        detail.setGoodsEntityId(entity.getId());
        detail.setSalePrice(entity.getSalePrice());
        detail.setMarketPrice(entity.getMarketPrice());
        detail.setInventory(entity.getInventory() != null ? entity.getInventory() : 0);
        detail.setGoodsArtno(entity.getProductArtno());
    }

    private void initCollections(ProductDetailDTO detail) {
        detail.setPictures(new ArrayList<>());
        detail.setAttributes(new ArrayList<>());
        detail.setSpecifications(new ArrayList<>());
        detail.setBreadcrumbs(new ArrayList<>());
        detail.setComments(new ArrayList<>());
    }

    private Double convertScore(BigDecimal score) {
        return score != null ? score.doubleValue() : 0.0;
    }

    public List<ProductDTO> listProducts(Integer categoryId, String keyword, String priceRange) {
        QueryWrapper<ProductDO> query = buildListQuery(categoryId, keyword);
        List<ProductDO> items = productMapper.selectList(query);
        if (items.isEmpty()) {
            return new ArrayList<>();
        }

        // 批量查询，修复 N+1 问题
        List<Integer> productIds = items.stream().map(ProductDO::getId).collect(Collectors.toList());
        Map<Integer, ProductStatsDO> statsMap = batchGetStats(productIds);
        Map<Integer, ProductEntityDO> defaultSkuMap = batchGetDefaultSkus(productIds);

        return items.stream()
                .map(item -> convertToProductDTO(item, statsMap.get(item.getId()), defaultSkuMap.get(item.getId())))
                .collect(Collectors.toList());
    }

    private QueryWrapper<ProductDO> buildListQuery(Integer categoryId, String keyword) {
        QueryWrapper<ProductDO> query = new QueryWrapper<>();
        if (categoryId != null && categoryId != 0) {
            // TODO: implement category filtering
        }
        if (keyword != null && !keyword.isEmpty()) {
            query.like("product_name", keyword);
        }
        return query;
    }

    // 批量查询 ProductStats
    private Map<Integer, ProductStatsDO> batchGetStats(List<Integer> productIds) {
        if (productIds.isEmpty()) {
            return new HashMap<>();
        }
        List<ProductStatsDO> statsList = productStatsMapper.selectBatchIds(productIds);
        return statsList.stream().collect(Collectors.toMap(ProductStatsDO::getProductId, Function.identity()));
    }

    // 批量查询 ProductEntity 并找出每个产品的最低价 SKU
    private Map<Integer, ProductEntityDO> batchGetDefaultSkus(List<Integer> productIds) {
        if (productIds.isEmpty()) {
            return new HashMap<>();
        }
        List<ProductEntityDO> entities = productEntityMapper.selectList(
                new QueryWrapper<ProductEntityDO>().in("product_id", productIds));

        // 按 productId 分组，找出每个产品的最低价 SKU
        Map<Integer, List<ProductEntityDO>> grouped = entities.stream()
                .collect(Collectors.groupingBy(ProductEntityDO::getProductId));

        Map<Integer, ProductEntityDO> result = new HashMap<>();
        for (Map.Entry<Integer, List<ProductEntityDO>> entry : grouped.entrySet()) {
            entry.getValue().stream()
                    .min(Comparator.comparing(ProductEntityDO::getSalePrice))
                    .ifPresent(entity -> result.put(entry.getKey(), entity));
        }
        return result;
    }

    private ProductDTO convertToProductDTO(ProductDO item, ProductStatsDO stats, ProductEntityDO defaultSku) {
        ProductDTO dto = new ProductDTO();
        dto.setId(item.getId());
        dto.setName(item.getProductName());
        dto.setGoodsNo(item.getProductNo());

        // 使用已批量查询的数据
        if (stats != null) {
            dto.setCommentNumber(stats.getCommentNumber());
            dto.setScores(convertScore(stats.getScores()));
        }
        if (defaultSku != null) {
            dto.setPrice(defaultSku.getSalePrice() != null ? defaultSku.getSalePrice().toString() : null);
            dto.setMarketPrice(defaultSku.getMarketPrice() != null ? defaultSku.getMarketPrice().toString() : null);
            dto.setImageUrl(defaultSku.getImageUrl());
        }

        return dto;
    }

    // 以下方法保留用于单产品详情查询（已优化）
    private void enrichWithStats(ProductDTO dto, Integer productId) {
        ProductStatsDO stats = productStatsMapper.selectById(productId);
        if (stats != null) {
            dto.setCommentNumber(stats.getCommentNumber());
            dto.setScores(convertScore(stats.getScores()));
        }
    }

    private void enrichWithDefaultSku(ProductDTO dto, Integer productId) {
        List<ProductEntityDO> entities = productEntityMapper.selectList(
                new QueryWrapper<ProductEntityDO>().eq("product_id", productId));
        entities.stream()
                .min(Comparator.comparing(ProductEntityDO::getSalePrice))
                .ifPresent(entity -> {
                    dto.setPrice(entity.getSalePrice() != null ? entity.getSalePrice().toString() : null);
                    dto.setMarketPrice(entity.getMarketPrice() != null ? entity.getMarketPrice().toString() : null);
                    dto.setImageUrl(entity.getImageUrl());
                });
    }

    public void createProduct() {
        // TODO: Map to domain entity and persist
        Long id = IdWorker.getId();
    }

    public void updateProduct(Long productId, byte[] description) {
        // TODO: Load, update and persist
    }

    public void deleteProduct(String productId) {
        String eventMessage = "delete product with:" + productId;
        sendProductEvent(ProductEventType.PRODUCT_DELETED, eventMessage);
    }

    private void sendProductEvent(ProductEventType eventType, String message) {
        try {
            mqProducer.send(
                    PRODUCT_EVENT_TOPIC,
                    message,
                    eventType.getEventName(),
                    null);
        } catch (Exception e) {
            throw new ProductDomainException(
                    ProductErrorCode.DELETE_FAILED,
                    "Failed to send product event: " + eventType.getEventName(),
                    e);
        }
    }

    public void uploadImage(Long productId, MultipartFile imageFile) {
        String imageId = String.valueOf(IdWorker.getId());
        saveImage(productId, imageId);
    }

    private void saveImage(Long productId, String imageUrl) {
        String imageId = String.valueOf(IdWorker.getId());
        productImageService.create(productId, imageId, imageUrl);
    }

    public ProductDTO getProductById(Integer id) {
        ProductDO product = productMapper.selectById(id);
        if (product == null) {
            return null;
        }
        // 批量查询数据
        ProductStatsDO stats = productStatsMapper.selectById(id);
        List<ProductEntityDO> entities = productEntityMapper.selectList(
                new QueryWrapper<ProductEntityDO>().eq("product_id", id));
        ProductEntityDO defaultSku = entities.stream()
                .min(Comparator.comparing(ProductEntityDO::getSalePrice))
                .orElse(null);
        return convertToProductDTO(product, stats, defaultSku);
    }

    public List<ProductDTO> searchProducts(String keyword, Integer categoryId, Integer brandId, Integer sort,
            Integer pageNum, Integer pageSize) {
        QueryWrapper<ProductDO> query = new QueryWrapper<>();

        if (keyword != null && !keyword.isEmpty()) {
            query.and(wrapper -> wrapper
                    .like("product_name", keyword)
                    .or()
                    .like("product_no", keyword)
                    .or()
                    .like("product_remark", keyword));
        }

        if (brandId != null) {
            query.eq("brand_id", brandId);
        }

        // 排序逻辑
        switch (sort) {
            case 1: // 销量
                query.orderByDesc("sale_count");
                break;
            case 2: // 价格升序
                query.orderByAsc("sale_price");
                break;
            case 3: // 价格降序
                query.orderByDesc("sale_price");
                break;
            default: // 综合
                query.orderByDesc("create_time");
                break;
        }

        // 分页
        int offset = (pageNum - 1) * pageSize;
        query.last("LIMIT " + pageSize + " OFFSET " + offset);

        List<ProductDO> items = productMapper.selectList(query);
        if (items.isEmpty()) {
            return new ArrayList<>();
        }

        // 批量查询，修复 N+1 问题
        List<Integer> productIds = items.stream().map(ProductDO::getId).collect(Collectors.toList());
        Map<Integer, ProductStatsDO> statsMap = batchGetStats(productIds);
        Map<Integer, ProductEntityDO> defaultSkuMap = batchGetDefaultSkus(productIds);

        return items.stream()
                .map(item -> convertToProductDTO(item, statsMap.get(item.getId()), defaultSkuMap.get(item.getId())))
                .collect(Collectors.toList());
    }

    public List<ProductDTO> simpleSearch(String keyword, Integer limit) {
        QueryWrapper<ProductDO> query = new QueryWrapper<>();
        query.and(wrapper -> wrapper
                .like("product_name", keyword)
                .or()
                .like("product_no", keyword));
        query.last("LIMIT " + limit);

        List<ProductDO> items = productMapper.selectList(query);
        if (items.isEmpty()) {
            return new ArrayList<>();
        }

        // 批量查询，修复 N+1 问题
        List<Integer> productIds = items.stream().map(ProductDO::getId).collect(Collectors.toList());
        Map<Integer, ProductStatsDO> statsMap = batchGetStats(productIds);
        Map<Integer, ProductEntityDO> defaultSkuMap = batchGetDefaultSkus(productIds);

        return items.stream()
                .map(item -> convertToProductDTO(item, statsMap.get(item.getId()), defaultSkuMap.get(item.getId())))
                .collect(Collectors.toList());
    }

    public List<ProductDTO> recommendProducts(Integer productId, Integer limit) {
        ProductDO product = productMapper.selectById(productId);
        if (product == null) {
            return new ArrayList<>();
        }

        // 基于产品名称前缀推荐（简化实现）
        QueryWrapper<ProductDO> query = new QueryWrapper<>();
        query.ne("id", productId)
                .orderByDesc("create_time")
                .last("LIMIT " + limit);

        List<ProductDO> items = productMapper.selectList(query);
        if (items.isEmpty()) {
            return new ArrayList<>();
        }
        return items.stream()
                .map(item -> convertToProductDTO(item, null, null))
                .collect(Collectors.toList());
    }
}
