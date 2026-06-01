package com.metawebthree.product_recommendation.infrastructure.persistence.repository;

import com.metawebthree.product_recommendation.domain.model.ProductProfile;
import com.metawebthree.product_recommendation.domain.repository.ProductProfileRepository;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.Collectors;

public class ProductProfileRepositoryImpl implements ProductProfileRepository {

    private final Map<Long, ProductProfile> storage = new ConcurrentHashMap<>();
    private Long idCounter = 1L;

    @Override
    public void save(ProductProfile profile) {
        if (profile.getId() == null) {
            profile.setId(idCounter++);
        }
        storage.put(profile.getProductId(), profile);
    }

    @Override
    public void batchSave(List<ProductProfile> profiles) {
        for (ProductProfile profile : profiles) {
            save(profile);
        }
    }

    @Override
    public ProductProfile findByProductId(Long productId) {
        return storage.get(productId);
    }

    @Override
    public List<ProductProfile> findByProductIds(List<Long> productIds) {
        return productIds.stream()
                .map(storage::get)
                .filter(p -> p != null)
                .collect(Collectors.toList());
    }

    @Override
    public List<ProductProfile> findByCategory(String category) {
        return storage.values().stream()
                .filter(p -> category.equals(p.getCategory()))
                .collect(Collectors.toList());
    }

    @Override
    public void updateSimilarProducts(Long productId, List<Long> similarIds) {
        ProductProfile profile = storage.get(productId);
        if (profile != null) {
            profile.setSimilarProductIds(similarIds);
        }
    }
}