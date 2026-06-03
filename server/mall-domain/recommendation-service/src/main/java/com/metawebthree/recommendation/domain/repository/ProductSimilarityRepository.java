package com.metawebthree.recommendation.domain.repository;

import com.metawebthree.recommendation.domain.entity.ProductSimilarity;
import java.time.LocalDateTime;
import java.util.List;

public interface ProductSimilarityRepository {
    ProductSimilarity save(ProductSimilarity productSimilarity);
    List<ProductSimilarity> findSimilarProducts(Long productId);
    ProductSimilarity findByProductIds(Long productId1, Long productId2);
    List<ProductSimilarity> findSimilarProductsByAlgorithm(Long productId, ProductSimilarity.SimilarityAlgorithm algorithm);
    void deleteByLastUpdatedBefore(LocalDateTime timestamp);
    boolean existsSimilarity(Long productId1, Long productId2);
    List<ProductSimilarity> findAll();
    void deleteById(Long id);
}
