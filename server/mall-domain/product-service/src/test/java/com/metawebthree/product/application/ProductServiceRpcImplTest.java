package com.metawebthree.product.application;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.*;
import static org.mockito.Mockito.*;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.baomidou.mybatisplus.core.metadata.IPage;
import com.metawebthree.common.generated.rpc.*;
import com.metawebthree.product.domain.model.ProductDO;
import com.metawebthree.product.domain.model.ProductEntityDO;
import com.metawebthree.product.dto.ProductDetailDTO;
import com.metawebthree.product.infrastructure.persistence.mapper.ProductEntityMapper;
import com.metawebthree.product.infrastructure.persistence.mapper.ProductMapper;

import java.math.BigDecimal;
import java.time.LocalDateTime;
import java.util.List;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.ArgumentCaptor;
import org.mockito.Captor;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

@ExtendWith(MockitoExtension.class)
class ProductServiceRpcImplTest {

    @Mock
    private ProductService productService;
    @Mock
    private ProductMapper productMapper;
    @Mock
    private ProductEntityMapper productEntityMapper;

    @InjectMocks
    private ProductServiceRpcImpl productServiceRpc;

    @Captor
    private ArgumentCaptor<ProductDO> productCaptor;
    @Captor
    private ArgumentCaptor<ProductEntityDO> entityCaptor;

    @Test
    void getProductDetail_whenExists_shouldReturnProduct() {
        int productId = 1;
        ProductDetailDTO detail = new ProductDetailDTO();
        detail.setId(productId);
        detail.setGoodsName("Test Product");
        detail.setImageUrl("http://example.com/pic.jpg");
        detail.setGoodsRemark("A great product");
        detail.setSalePrice(BigDecimal.valueOf(99.99));
        detail.setPictures(List.of("http://example.com/pic1.jpg"));

        when(productService.getProductDetail(productId)).thenReturn(detail);

        GetProductDetailRequest request = GetProductDetailRequest.newBuilder()
                .setProductId(productId).build();

        GetProductDetailResponse response = productServiceRpc.getProductDetail(request);

        assertTrue(response.hasProduct());
        assertEquals(productId, response.getProduct().getId());
        assertEquals("Test Product", response.getProduct().getName());
        assertEquals("http://example.com/pic.jpg", response.getProduct().getPic());
        assertEquals("A great product", response.getProduct().getSubTitle());
        assertEquals(99.99, response.getProduct().getPrice(), 0.001);
        assertEquals(1, response.getProduct().getPicturesCount());
        assertEquals("http://example.com/pic1.jpg", response.getProduct().getPictures(0));
    }

    @Test
    void getProductDetail_whenNotFound_shouldReturnEmpty() {
        when(productService.getProductDetail(anyInt())).thenReturn(null);

        GetProductDetailRequest request = GetProductDetailRequest.newBuilder()
                .setProductId(999).build();

        GetProductDetailResponse response = productServiceRpc.getProductDetail(request);

        assertFalse(response.hasProduct());
    }

    @Test
    void listProducts_shouldReturnPaginatedResults() {
        int page = 1;
        int size = 10;

        ProductDO product = new ProductDO();
        product.setId(1);
        product.setProductName("Test Product");
        product.setProductRemark("Remark");
        product.setCreateTime(LocalDateTime.now());

        IPage<ProductDO> pageResult = new Page<>(page, size, 1);
        pageResult.setRecords(List.of(product));

        when(productMapper.selectPage(any(Page.class), any(LambdaQueryWrapper.class)))
            .thenReturn(pageResult);

        ListProductsRequest request = ListProductsRequest.newBuilder()
                .setPage(page).setSize(size).setCategoryId(0L).build();

        ListProductsResponse response = productServiceRpc.listProducts(request);

        assertEquals(1, response.getTotalCount());
        assertEquals(page, response.getPage());
        assertEquals(size, response.getSize());
        assertEquals(1, response.getProductsCount());
        assertEquals(1, response.getProducts(0).getId());
        assertEquals("Test Product", response.getProducts(0).getName());
    }

    @Test
    void listProducts_withCategoryFilter_shouldFilterByCategory() {
        ProductDO product = new ProductDO();
        product.setId(1);
        product.setProductName("Categorized Product");
        product.setCreateTime(LocalDateTime.now());

        IPage<ProductDO> pageResult = new Page<>(1, 10, 1);
        pageResult.setRecords(List.of(product));

        when(productMapper.selectPage(any(Page.class), any(LambdaQueryWrapper.class)))
            .thenReturn(pageResult);

        ListProductsRequest request = ListProductsRequest.newBuilder()
                .setPage(1).setSize(10).setCategoryId(5L).build();

        ListProductsResponse response = productServiceRpc.listProducts(request);

        assertEquals(1, response.getProductsCount());
    }

    @Test
    void createProduct_shouldCreateSuccessfully() {
        CreateProductRequest request = CreateProductRequest.newBuilder()
                .setName("New Product")
                .setSku("SKU-001")
                .setSubTitle("A new product")
                .setCategoryId(10L)
                .setPrice(49.99)
                .setStock(100)
                .setPic("http://example.com/pic.jpg")
                .build();

        when(productMapper.insert(any(ProductDO.class))).thenReturn(1);
        when(productEntityMapper.insert(any(ProductEntityDO.class))).thenReturn(1);

        CreateProductResponse response = productServiceRpc.createProduct(request);

        assertTrue(response.getSuccess());

        verify(productMapper).insert(productCaptor.capture());
        ProductDO savedProduct = productCaptor.getValue();
        assertEquals("New Product", savedProduct.getProductName());
        assertEquals("SKU-001", savedProduct.getProductNo());
        assertEquals("A new product", savedProduct.getProductRemark());
        assertEquals(10L, savedProduct.getCategoryId().longValue());
        assertNotNull(savedProduct.getCreateTime());

        verify(productEntityMapper).insert(entityCaptor.capture());
        ProductEntityDO savedEntity = entityCaptor.getValue();
        assertEquals("SKU-001", savedEntity.getSku());
        assertEquals(BigDecimal.valueOf(49.99), savedEntity.getSalePrice());
        assertEquals(100, savedEntity.getInventory().intValue());
        assertEquals("http://example.com/pic.jpg", savedEntity.getImageUrl());
    }
}
