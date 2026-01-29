package com.metawebthree.product;

import com.baomidou.mybatisplus.core.conditions.query.QueryWrapper;
import com.baomidou.mybatisplus.core.toolkit.IdWorker;
import com.metawebthree.common.dto.ProductDTO;
import com.metawebthree.common.dto.ProductDetailDTO;
import com.metawebthree.common.utils.RocketMQ.MQProducer;
import com.metawebthree.image.ProductImageService;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.stream.Collectors;

import lombok.extern.slf4j.Slf4j;

import org.apache.rocketmq.client.exception.MQBrokerException;
import org.apache.rocketmq.client.exception.MQClientException;
import org.apache.rocketmq.remoting.exception.RemotingException;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;

import java.util.concurrent.ConcurrentHashMap;

@Service
@Slf4j
public class ProductService {
    private final MQProducer mqProducer;
    private final ProductImageService productImageService;
    private final GoodsMapper goodsMapper;
    private final GoodsEntityMapper goodsEntityMapper;
    private final GoodsStatsMapper goodsStatsMapper;
    private final GoodsLimitsMapper goodsLimitsMapper;

    public ProductService(MQProducer mqProducer, ProductImageService productImageService, 
                          GoodsMapper goodsMapper, GoodsEntityMapper goodsEntityMapper,
                          GoodsStatsMapper goodsStatsMapper, GoodsLimitsMapper goodsLimitsMapper) {
        this.mqProducer = mqProducer;
        this.productImageService = productImageService;
        this.goodsMapper = goodsMapper;
        this.goodsEntityMapper = goodsEntityMapper;
        this.goodsStatsMapper = goodsStatsMapper;
        this.goodsLimitsMapper = goodsLimitsMapper;
    }

    public ProductDetailDTO getProductDetail(Integer id) {
        GoodsDO goods = goodsMapper.selectById(id);
        if (goods == null) return null;

        List<GoodsEntityDO> entities = goodsEntityMapper.selectList(
            new QueryWrapper<GoodsEntityDO>().eq("goods_id", id)
        );

        GoodsEntityDO defaultEntity = entities.stream()
            .min(Comparator.comparing(GoodsEntityDO::getSalePrice))
            .orElse(new GoodsEntityDO());

        GoodsStatsDO stats = goodsStatsMapper.selectById(id);
        GoodsLimitsDO limits = goodsLimitsMapper.selectById(id);

        ProductDetailDTO detail = new ProductDetailDTO();
        detail.setId(goods.getId());
        detail.setGoodsName(goods.getGoodsName());
        detail.setGoodsNo(goods.getGoodsNo());
        detail.setGoodsRemark(goods.getGoodsRemark());
        
        if (stats != null) {
            detail.setCommentNumber(stats.getCommentNumber());
            detail.setScoreNumber(stats.getScoreNumber());
            detail.setScores(stats.getScores() != null ? stats.getScores().doubleValue() : 0.0);
        }
        
        if (limits != null) {
            detail.setPurchase(limits.getPurchase());
        }

        detail.setGoodsEntityId(defaultEntity.getId());
        detail.setSalePrice(defaultEntity.getSalePrice());
        detail.setMarketPrice(defaultEntity.getMarketPrice());
        detail.setInventory(defaultEntity.getInventory());
        detail.setGoodsArtno(defaultEntity.getGoodsArtno());

        // Stubs for complex dependencies
        detail.setPictures(new ArrayList<>()); 
        detail.setAttributes(new ArrayList<>());
        detail.setSpecifications(new ArrayList<>());
        detail.setBreadcrumbs(new ArrayList<>());
        detail.setComments(new ArrayList<>());

        return detail;
    }

    public List<ProductDTO> listProducts(Integer categoryId, String keyword, String priceRange) {
        // Simple mock implementation of filtered listing
        QueryWrapper<GoodsDO> query = new QueryWrapper<>();
        if (categoryId != null && categoryId != 0) {
            // In a real scenario, this would involve a join with tb_goods_category_mapping
        }
        if (keyword != null && !keyword.isEmpty()) {
            query.like("goods_name", keyword);
        }
        
        List<GoodsDO> items = goodsMapper.selectList(query);
        return items.stream().map(item -> {
            ProductDTO dto = new ProductDTO();
            dto.setId(item.getId());
            dto.setName(item.getGoodsName());
            dto.setGoodsNo(item.getGoodsNo());
            
            // Fetch default SKU for price and image
            List<GoodsEntityDO> entities = goodsEntityMapper.selectList(
                new QueryWrapper<GoodsEntityDO>().eq("goods_id", item.getId())
            );
            entities.stream()
                .min(Comparator.comparing(GoodsEntityDO::getSalePrice))
                .ifPresent(entity -> {
                    dto.setPrice(entity.getSalePrice().toString());
                    dto.setMarketPrice(entity.getMarketPrice().toString());
                    dto.setImageUrl(entity.getImageUrl());
                });

            GoodsStatsDO stats = goodsStatsMapper.selectById(item.getId());
            if (stats != null) {
                dto.setCommentNumber(stats.getCommentNumber());
                dto.setScores(stats.getScores() != null ? stats.getScores().doubleValue() : 0.0);
            }
            
            return dto;
        }).collect(Collectors.toList());
    }

    public Boolean createProduct() {
        Long id = IdWorker.getId();
        // @TODO sql
        return Boolean.valueOf(true);
    }

    public boolean updateProduct(Long id, byte[] description) {
        // @TODO sql modify
        return true;
    }

    public void deleteProduct(String key)
            throws MQBrokerException, RemotingException, InterruptedException, MQClientException {
        mqProducer.send("deleteProduct", "delete product with:" + key, null, null);
    }

    public boolean uploadImage(Long productId, MultipartFile imageFile) {
        String imageId = String.valueOf(IdWorker.getId());
        // @TODO upload image with MediaService
        return saveImage(productId, imageId);
    }

    public boolean saveImage(Long productId, String imageUrl) {
        String imageId = String.valueOf(IdWorker.getId());
        return Integer.valueOf(1).equals(productImageService.create(productId, imageId, imageUrl));
    }

    ConcurrentHashMap<Long, Object> statisticLockMap = new ConcurrentHashMap<>();
    Long count = 0L;

    public boolean updateFeatureStatistic(Long featureId, Integer increated) {
        return updateFeatureStatisticWithLock(featureId, increated);
    }

    public boolean updateFeatureStatisticWithLock(Long featureId, Integer increated) {
        Object lock = statisticLockMap.computeIfAbsent(featureId, key -> new Object());
        synchronized (lock) {
            count += increated;
        }
        return true;
    }
}
