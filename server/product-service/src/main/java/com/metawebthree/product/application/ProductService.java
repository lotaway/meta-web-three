package com.metawebthree.product.application;

import com.baomidou.mybatisplus.core.conditions.query.QueryWrapper;
import com.baomidou.mybatisplus.core.toolkit.IdWorker;
import com.metawebthree.common.dto.ProductDTO;
import com.metawebthree.common.dto.ProductDetailDTO;
import com.metawebthree.common.utils.RocketMQ.MQProducer;
import com.metawebthree.image.ProductImageService;
import com.metawebthree.product.domain.event.ProductEventType;
import com.metawebthree.product.domain.exception.ProductDomainException;
import com.metawebthree.product.domain.exception.ProductErrorCode;
import com.metawebthree.product.domain.model.*;
import com.metawebthree.product.infrastructure.persistence.mapper.*;
import java.math.BigDecimal;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.stream.Collectors;

import lombok.extern.slf4j.Slf4j;

import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;

@Service
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
        if (stats == null) return;
        detail.setCommentNumber(stats.getCommentNumber());
        detail.setScoreNumber(stats.getScoreNumber());
        detail.setScores(convertScore(stats.getScores()));
    }

    private void mapLimitsInfo(ProductDetailDTO detail, ProductLimitsDO limits) {
        if (limits == null) return;
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
        return items.stream()
                .map(this::convertToProductDTO)
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

    private ProductDTO convertToProductDTO(ProductDO item) {
        ProductDTO dto = new ProductDTO();
        dto.setId(item.getId());
        dto.setName(item.getProductName());
        dto.setGoodsNo(item.getProductNo());

        enrichWithDefaultSku(dto, item.getId());
        enrichWithStats(dto, item.getId());

        return dto;
    }

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
                    ProductErrorCode.PRODUCT_DELETE_FAILED,
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
        return convertToProductDTO(product);
    }
}
