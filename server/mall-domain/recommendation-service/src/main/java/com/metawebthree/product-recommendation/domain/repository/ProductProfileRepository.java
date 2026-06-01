package com.metawebthree.product_recommendation.domain.repository;

import com.metawebthree.product_recommendation.domain.model.ProductProfile;
import java.util.List;

public interface ProductProfileRepository {
    void save(ProductProfile profile);
    void batchSave(List<ProductProfile> profiles);
    ProductProfile findByProductId(Long productId);
    List<ProductProfile> findByProductIds(List<Long> productIds);
    List<ProductProfile> findByCategory(String category);
    void updateSimilarProducts(Long productId, List<Long> similarIds);
}