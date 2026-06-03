package com.metawebthree.recommendation.infrastructure.persistence.repository;

import com.metawebthree.recommendation.domain.entity.ProductSimilarity;
import com.metawebthree.recommendation.domain.repository.ProductSimilarityRepository;
import com.metawebthree.recommendation.infrastructure.persistence.entity.ProductSimilarityDO;
import com.metawebthree.recommendation.infrastructure.persistence.mapper.ProductSimilarityMapper;
import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import java.time.LocalDateTime;
import java.util.List;
import java.util.stream.Collectors;
import org.springframework.stereotype.Repository;

@Repository
public class ProductSimilarityRepositoryImpl implements ProductSimilarityRepository {

    private final ProductSimilarityMapper productSimilarityMapper;

    public ProductSimilarityRepositoryImpl(ProductSimilarityMapper productSimilarityMapper) {
        this.productSimilarityMapper = productSimilarityMapper;
    }

    @Override
    public ProductSimilarity save(ProductSimilarity productSimilarity) {
        ProductSimilarityDO productSimilarityDO = toDO(productSimilarity);
        if (productSimilarity.getId() == null) {
            productSimilarityMapper.insert(productSimilarityDO);
            productSimilarity.setId(productSimilarityDO.getId());
        } else {
            productSimilarityMapper.updateById(productSimilarityDO);
        }
        return productSimilarity;
    }

    @Override
    public List<ProductSimilarity> findSimilarProducts(Long productId) {
        LambdaQueryWrapper<ProductSimilarityDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.and(w -> w.eq(ProductSimilarityDO::getProductId1, productId)
            .or()
            .eq(ProductSimilarityDO::getProductId2, productId));
        return productSimilarityMapper.selectList(wrapper).stream()
            .map(this::toDomain)
            .collect(Collectors.toList());
    }

    @Override
    public ProductSimilarity findByProductIds(Long productId1, Long productId2) {
        LambdaQueryWrapper<ProductSimilarityDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.and(w -> w.and(w1 -> w1.eq(ProductSimilarityDO::getProductId1, productId1)
            .eq(ProductSimilarityDO::getProductId2, productId2))
            .or(w2 -> w2.eq(ProductSimilarityDO::getProductId1, productId2)
            .eq(ProductSimilarityDO::getProductId2, productId1)));
        ProductSimilarityDO productSimilarityDO = productSimilarityMapper.selectOne(wrapper);
        return productSimilarityDO != null ? toDomain(productSimilarityDO) : null;
    }

    @Override
    public List<ProductSimilarity> findSimilarProductsByAlgorithm(Long productId, ProductSimilarity.SimilarityAlgorithm algorithm) {
        LambdaQueryWrapper<ProductSimilarityDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.and(w -> w.eq(ProductSimilarityDO::getProductId1, productId)
            .or()
            .eq(ProductSimilarityDO::getProductId2, productId))
            .eq(ProductSimilarityDO::getAlgorithm, algorithm.name());
        return productSimilarityMapper.selectList(wrapper).stream()
            .map(this::toDomain)
            .collect(Collectors.toList());
    }

    @Override
    public void deleteByLastUpdatedBefore(LocalDateTime timestamp) {
        LambdaQueryWrapper<ProductSimilarityDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.lt(ProductSimilarityDO::getLastUpdated, timestamp);
        productSimilarityMapper.delete(wrapper);
    }

    @Override
    public boolean existsSimilarity(Long productId1, Long productId2) {
        LambdaQueryWrapper<ProductSimilarityDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.and(w -> w.and(w1 -> w1.eq(ProductSimilarityDO::getProductId1, productId1)
            .eq(ProductSimilarityDO::getProductId2, productId2))
            .or(w2 -> w2.eq(ProductSimilarityDO::getProductId1, productId2)
            .eq(ProductSimilarityDO::getProductId2, productId1)));
        return productSimilarityMapper.selectCount(wrapper) > 0;
    }

    @Override
    public List<ProductSimilarity> findAll() {
        return productSimilarityMapper.selectList(new LambdaQueryWrapper<>()).stream()
            .map(this::toDomain)
            .collect(Collectors.toList());
    }

    @Override
    public void deleteById(Long id) {
        productSimilarityMapper.deleteById(id);
    }

    private ProductSimilarity toDomain(ProductSimilarityDO productSimilarityDO) {
        ProductSimilarity productSimilarity = new ProductSimilarity();
        productSimilarity.setId(productSimilarityDO.getId());
        productSimilarity.setProductId1(productSimilarityDO.getProductId1());
        productSimilarity.setProductId2(productSimilarityDO.getProductId2());
        productSimilarity.setSimilarityScore(productSimilarityDO.getSimilarityScore());
        productSimilarity.setAlgorithm(ProductSimilarity.SimilarityAlgorithm.valueOf(productSimilarityDO.getAlgorithm()));
        productSimilarity.setLastUpdated(productSimilarityDO.getLastUpdated());
        productSimilarity.setUpdateCount(productSimilarityDO.getUpdateCount());
        return productSimilarity;
    }

    private ProductSimilarityDO toDO(ProductSimilarity productSimilarity) {
        ProductSimilarityDO productSimilarityDO = new ProductSimilarityDO();
        productSimilarityDO.setId(productSimilarity.getId());
        productSimilarityDO.setProductId1(productSimilarity.getProductId1());
        productSimilarityDO.setProductId2(productSimilarity.getProductId2());
        productSimilarityDO.setSimilarityScore(productSimilarity.getSimilarityScore());
        productSimilarityDO.setAlgorithm(productSimilarity.getAlgorithm().name());
        productSimilarityDO.setLastUpdated(productSimilarity.getLastUpdated());
        productSimilarityDO.setUpdateCount(productSimilarity.getUpdateCount());
        return productSimilarityDO;
    }
}
