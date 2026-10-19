package com.metawebthree.product.application;

import com.baomidou.mybatisplus.core.conditions.query.QueryWrapper;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.metawebthree.product.domain.model.BrandDO;
import com.metawebthree.product.domain.model.PrefrenceAreaDO;
import com.metawebthree.product.domain.model.ProductCategoryDO;
import com.metawebthree.product.domain.model.SubjectDO;
import com.metawebthree.product.dto.HomeContentDTO;
import com.metawebthree.product.dto.ProductDTO;
import com.metawebthree.product.infrastructure.persistence.mapper.BrandMapper;
import com.metawebthree.product.infrastructure.persistence.mapper.PrefrenceAreaMapper;
import com.metawebthree.product.infrastructure.persistence.mapper.ProductCategoryMapper;
import com.metawebthree.product.infrastructure.persistence.mapper.SubjectMapper;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;

import java.util.List;
import java.util.stream.Collectors;

/**
 * Assembles the shopper homepage from persisted operational content. Nothing here
 * is hard-coded: products come from {@link ProductService} and the brand,
 * category, subject and preference-area records come from their tables.
 */
@Service
@RequiredArgsConstructor
public class HomeContentService {

    private static final int DEFAULT_PAGE_SIZE = 10;

    private final ProductService productService;
    private final ProductCategoryMapper productCategoryMapper;
    private final SubjectMapper subjectMapper;
    private final PrefrenceAreaMapper prefrenceAreaMapper;
    private final BrandMapper brandMapper;

    public HomeContentDTO content() {
        return HomeContentDTO.builder()
                .newProductList(productService.listProducts(null, "new", null))
                .hotProductList(productService.listProducts(null, "hot", null))
                .brandList(listBrands())
                .advertiseList(listPreferenceAreas())
                .subjectList(subjectList(null, 1, DEFAULT_PAGE_SIZE))
                .homeFlashPromotion(null)
                .build();
    }

    public List<HomeContentDTO.CategoryDTO> productCateList(Long parentId) {
        long effectiveParentId = parentId == null ? 0L : parentId;
        QueryWrapper<ProductCategoryDO> wrapper = new QueryWrapper<>();
        wrapper.eq("parent_id", effectiveParentId);
        wrapper.orderByAsc("sort");
        return productCategoryMapper.selectList(wrapper).stream()
                .map(category -> HomeContentDTO.CategoryDTO.builder()
                        .id(category.getId())
                        .name(category.getName())
                        .parentId(category.getParentId())
                        .level(category.getLevel())
                        .icon(category.getIcon())
                        .build())
                .collect(Collectors.toList());
    }

    public List<HomeContentDTO.SubjectDTO> subjectList(Long categoryId, int pageNum, int pageSize) {
        QueryWrapper<SubjectDO> wrapper = new QueryWrapper<>();
        wrapper.eq("show_status", 1);
        if (categoryId != null) {
            wrapper.eq("category_id", categoryId);
        }
        wrapper.orderByDesc("id");

        Page<SubjectDO> page = subjectMapper.selectPage(new Page<>(pageNum, pageSize), wrapper);
        return page.getRecords().stream()
                .map(subject -> HomeContentDTO.SubjectDTO.builder()
                        .id(subject.getId())
                        .title(subject.getTitle())
                        .pic(subject.getPic())
                        .categoryId(subject.getCategoryId())
                        .build())
                .collect(Collectors.toList());
    }

    public List<ProductDTO> newProductList(int pageNum, int pageSize) {
        return productService.listProducts(null, "new", null, pageNum, pageSize);
    }

    public List<ProductDTO> hotProductList(int pageNum, int pageSize) {
        return productService.listProducts(null, "hot", null, pageNum, pageSize);
    }

    public List<ProductDTO> recommendProductList(int pageNum, int pageSize) {
        return productService.listProducts(null, null, null, pageNum, pageSize);
    }

    public List<HomeContentDTO.BrandDTO> listBrands() {
        QueryWrapper<BrandDO> wrapper = new QueryWrapper<>();
        wrapper.orderByAsc("sort");
        return brandMapper.selectList(wrapper).stream()
                .map(brand -> HomeContentDTO.BrandDTO.builder()
                        .id(brand.getId())
                        .name(brand.getName())
                        .logo(brand.getLogo())
                        .productCount(brand.getProductCount())
                        .build())
                .collect(Collectors.toList());
    }

    /** Homepage operational banners, driven by the persisted preference areas. */
    public List<HomeContentDTO.AdvertiseDTO> listPreferenceAreas() {
        QueryWrapper<PrefrenceAreaDO> wrapper = new QueryWrapper<>();
        wrapper.eq("show_status", 1);
        wrapper.orderByAsc("sort");
        return prefrenceAreaMapper.selectList(wrapper).stream()
                .map(area -> HomeContentDTO.AdvertiseDTO.builder()
                        .id(area.getId())
                        .name(area.getName())
                        .pic(area.getPic())
                        .build())
                .collect(Collectors.toList());
    }
}
