package com.metawebthree.product.domain.repository;

import com.metawebthree.product.domain.model.ProductCategory;
import java.util.List;

public interface ProductCategoryRepository {
    void save(ProductCategory category);
    void update(ProductCategory category);
    ProductCategory findById(Long id);
    List<ProductCategory> findByParentId(Long parentId);
    void delete(Long id);
    List<ProductCategory> findAll();
}
