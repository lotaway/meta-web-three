package com.metawebthree.product.application;

import com.baomidou.mybatisplus.core.conditions.query.QueryWrapper;
import com.metawebthree.image.ProductImageService;
import com.metawebthree.common.utils.RocketMQ.MQProducer;
import com.metawebthree.product.domain.model.ProductCategoryDO;
import com.metawebthree.product.domain.model.ProductDO;
import com.metawebthree.product.dto.ProductDTO;
import com.metawebthree.product.infrastructure.persistence.mapper.ProductCategoryMapper;
import com.metawebthree.product.infrastructure.persistence.mapper.ProductEntityMapper;
import com.metawebthree.product.infrastructure.persistence.mapper.ProductLimitsMapper;
import com.metawebthree.product.infrastructure.persistence.mapper.ProductMapper;
import com.metawebthree.product.infrastructure.persistence.mapper.ProductStatsMapper;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.mockito.ArgumentCaptor;

import java.util.HashSet;
import java.util.List;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

/**
 * Category filtering must include a category's descendants and nothing else, and
 * must only return published, non-deleted products.
 */
class ProductCategoryFilterTest {

    private ProductCategoryMapper productCategoryMapper;
    private ProductMapper productMapper;
    private ProductService service;

    @BeforeEach
    void setUp() {
        productCategoryMapper = mock(ProductCategoryMapper.class);
        productMapper = mock(ProductMapper.class);
        ProductEntityMapper productEntityMapper = mock(ProductEntityMapper.class);
        ProductStatsMapper productStatsMapper = mock(ProductStatsMapper.class);
        when(productStatsMapper.selectBatchIds(any())).thenReturn(List.of());
        when(productEntityMapper.selectList(any(QueryWrapper.class))).thenReturn(List.of());
        service = new ProductService(
                mock(MQProducer.class),
                mock(ProductImageService.class),
                productMapper,
                productEntityMapper,
                productStatsMapper,
                mock(ProductLimitsMapper.class),
                productCategoryMapper);
    }

    private ProductCategoryDO category(Long id, Long parentId) {
        ProductCategoryDO category = new ProductCategoryDO();
        category.setId(id);
        category.setParentId(parentId);
        return category;
    }

    private void givenCategoryTree() {
        when(productCategoryMapper.selectList(null)).thenReturn(List.of(
                category(1L, 0L),
                category(2L, 1L),
                category(3L, 2L),
                category(5L, 0L),
                category(4L, 5L)));
    }

    @Test
    void categoryFilterIncludesOnlyItsOwnDescendants() {
        givenCategoryTree();

        List<Long> ids = service.resolveCategoryIds(1L);

        assertEquals(Set.of(1L, 2L, 3L), new HashSet<>(ids));
    }

    @Test
    void unknownCategoryMatchesNothing() {
        givenCategoryTree();

        List<Long> ids = service.resolveCategoryIds(99L);

        assertEquals(List.of(99L), ids);
    }

    @Test
    void listProductsAppliesCategoryAndPublishFilters() {
        givenCategoryTree();
        ProductDO product = new ProductDO();
        product.setId(10);
        product.setProductName("phone");
        when(productMapper.selectList(any(QueryWrapper.class))).thenReturn(List.of(product));

        List<ProductDTO> result = service.listProducts(1, null, null);

        assertEquals(1, result.size());
        ArgumentCaptor<QueryWrapper<ProductDO>> captor = ArgumentCaptor.forClass(QueryWrapper.class);
        verify(productMapper).selectList(captor.capture());
        String segment = captor.getValue().getSqlSegment();
        assertTrue(segment.contains("product_category_id"), segment);
        assertTrue(segment.contains("publish_status"), segment);
        assertTrue(segment.contains("delete_status"), segment);
        assertTrue(segment.contains("LIMIT"), segment);
    }

    @Test
    void listProductsWithoutCategoryDoesNotFilterByCategory() {
        when(productMapper.selectList(any(QueryWrapper.class))).thenReturn(List.of());

        service.listProducts(null, "new", null);

        ArgumentCaptor<QueryWrapper<ProductDO>> captor = ArgumentCaptor.forClass(QueryWrapper.class);
        verify(productMapper).selectList(captor.capture());
        assertFalse(captor.getValue().getSqlSegment().contains("product_category_id"));
        verify(productCategoryMapper, never()).selectList(any());
    }

    @Test
    void searchProductsFiltersByCategoryAndPublishedProducts() {
        givenCategoryTree();
        ProductDO product = new ProductDO();
        product.setId(11);
        product.setProductName("laptop");
        when(productMapper.selectList(any(QueryWrapper.class))).thenReturn(List.of(product));

        service.searchProducts("laptop", 1, null, 0, 1, 20);

        ArgumentCaptor<QueryWrapper<ProductDO>> captor = ArgumentCaptor.forClass(QueryWrapper.class);
        verify(productMapper).selectList(captor.capture());
        String segment = captor.getValue().getSqlSegment();
        assertTrue(segment.contains("product_category_id"), segment);
        assertTrue(segment.contains("publish_status"), segment);
    }

    @Test
    void keywordFilterUsesProductNameColumn() {
        when(productMapper.selectList(any(QueryWrapper.class))).thenReturn(List.of());

        service.listProducts(null, "phone", null);

        ArgumentCaptor<QueryWrapper<ProductDO>> captor = ArgumentCaptor.forClass(QueryWrapper.class);
        verify(productMapper).selectList(captor.capture());
        assertTrue(captor.getValue().getSqlSegment().contains("name"));
    }
}
