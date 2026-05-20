package com.metawebthree.product.application;

import com.metawebthree.product.domain.model.ProductCategory;
import com.metawebthree.product.domain.repository.ProductCategoryRepository;
import com.metawebthree.product.interfaces.web.dto.ProductCategoryNode;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;
import java.util.List;
import java.util.stream.Collectors;

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

    public List<ProductCategoryNode> categoryTreeList() {
        // 获取所有分类，并构造成树形结构
        List<ProductCategory> allCategories = productCategoryRepository.findAll();
        return allCategories.stream()
                .filter(ProductCategory::isRoot)
                .map(category -> buildNode(category, allCategories))
                .collect(Collectors.toList());
    }

    private ProductCategoryNode buildNode(ProductCategory category, List<ProductCategory> all) {
        ProductCategoryNode node = new ProductCategoryNode(category);
        List<ProductCategoryNode> children = all.stream()
                .filter(c -> category.getId().equals(c.getParentId()))
                .map(c -> buildNode(c, all))
                .collect(Collectors.toList());
        node.setChildren(children);
        return node;
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
