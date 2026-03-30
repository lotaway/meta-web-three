package com.metawebthree.product.domain.model;

import lombok.Builder;
import lombok.Getter;

@Getter
@Builder
public class Brand {
    private final Long id;
    private final String name;
    private final String firstLetter;
    private final Integer sort;
    private final boolean manufacturer;
    private final boolean visible;
    private final Integer productCount;
    private final Integer commentCount;
    private final String logo;
    private final String areaPicture;
    private final String story;
}
