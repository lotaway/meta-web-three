package com.metawebthree.product.domain.model;

import lombok.Builder;
import lombok.Getter;

@Getter
@Builder
public class Attribute {
    private final Long id;
    private final Long categoryId;
    private final String name;
    private final Integer selectType;
    private final Integer inputType;
    private final String options;
    private final Integer sort;
    private final Integer filterType;
    private final Integer searchType;
    private final boolean related;
    private final boolean supportsHandAdding;
    private final Integer attributeType;

    public boolean isSpecification() {
        return attributeType != null && attributeType == 0;
    }

    public boolean isParameter() {
        return attributeType != null && attributeType == 1;
    }
}
