package com.metawebthree.product.application;

import com.metawebthree.product.domain.model.ProductCategory;
import com.metawebthree.product.domain.repository.ProductCategoryRepository;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;
import java.util.List;

@Service
@RequiredArgsConstructor
public class ProductCategoryApplicationService {

    private final ProductCategoryRepository productCategoryRepository;

    public void createCategory(ProductCategory category) {
        validateCategory(category);
        productCategoryRepository.save(category);
    }

    public List<ProductCategory> findSubCategories(Long parentId) {
        return productCategoryRepository.findByParentId(parentId);
    }

    public void updateCategory(ProductCategory category) {
        validateId(category.getId());
        productCategoryRepository.update(category);
    }

    public void removeCategory(Long id) {
        validateId(id);
        productCategoryRepository.delete(id);
    }

    private void validateCategory(ProductCategory category) {
        if (category.getName() == null || category.getName().isEmpty()) {
            throw new IllegalArgumentException("Category name must not be empty");
        }
    }

    private void validateId(Long id) {
        if (id == null) {
            throw new IllegalArgumentException("Id must not be null");
        }
    }
}
