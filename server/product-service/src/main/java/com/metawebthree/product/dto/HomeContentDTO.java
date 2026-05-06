package com.metawebthree.product.dto;

import lombok.Builder;
import lombok.Getter;
import java.util.List;

@Getter
@Builder
public class HomeContentDTO {
    private List<AdvertiseDTO> advertiseList;
    private List<ProductDTO> newProductList;
    private List<ProductDTO> hotProductList;
    private List<BrandDTO> brandList;
    private List<SubjectDTO> subjectList;
    private FlashPromotionDTO homeFlashPromotion;

    @Getter
    @Builder
    public static class AdvertiseDTO {
        private Long id;
        private String name;
        private String pic;
        private String link;
    }

    @Getter
    @Builder
    public static class BrandDTO {
        private Long id;
        private String name;
        private String logo;
        private Integer productCount;
    }

    @Getter
    @Builder
    public static class SubjectDTO {
        private Long id;
        private String title;
        private String pic;
        private Long categoryId;
    }

    @Getter
    @Builder
    public static class FlashPromotionDTO {
        private String startTime;
        private String endTime;
        private List<ProductDTO> productList;
    }

    @Getter
    @Builder
    public static class CategoryDTO {
        private Long id;
        private String name;
        private Long parentId;
        private Integer level;
        private String icon;
    }
}
