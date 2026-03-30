package com.metawebthree.product.domain.model;

import lombok.Builder;
import lombok.Getter;

@Getter
@Builder
public class ProductCategory {
    private final Long id;
    private final Long parentId;
    private final String name;
    private final Integer level;
    private final Integer productCount;
    private final String productUnit;
    private final boolean displayedInNav;
    private final boolean visible;
    private final Integer sort;
    private final String icon;
    private final String keywords;
    private final String description;

    public boolean isRoot() {
        return parentId == null || parentId == 0;
    }
}
