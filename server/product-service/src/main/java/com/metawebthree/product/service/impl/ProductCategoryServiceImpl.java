package com.metawebthree.product.service.impl;

import com.metawebthree.product.domain.ProductCategory;
import com.metawebthree.product.service.ProductCategoryService;
import org.springframework.stereotype.Service;
import java.util.List;

@Service
public class ProductCategoryServiceImpl implements ProductCategoryService {
    @Override
    public List<ProductCategory> listCategories() {
        return List.of();
    }
}
