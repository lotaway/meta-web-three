package com.metawebthree.product.interfaces.web.dto;

import com.metawebthree.product.domain.model.ProductCategory;
import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Getter;
import lombok.Setter;

import java.util.List;

@Getter
@Setter
@Schema(description = "商品分类树节点")
public class ProductCategoryNode extends ProductCategory {
    @Schema(description = "子分类")
    private List<ProductCategoryNode> children;

    public ProductCategoryNode(ProductCategory category) {
        super(category.getId(), category.getParentId(), category.getName(),
                category.getLevel(), category.getProductCount(), category.getProductUnit(),
                category.isDisplayedInNav(), category.isVisible(), category.getSort(),
                category.getIcon(), category.getKeywords(), category.getDescription());
    }
}
