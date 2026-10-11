package com.metawebthree.product.application;

import java.util.List;
import java.util.concurrent.CompletableFuture;

import org.apache.dubbo.config.annotation.DubboService;

import com.metawebthree.common.generated.rpc.*;
import com.metawebthree.common.constants.PaginationConstants;
import com.metawebthree.common.utils.ValidationUtils;
import com.metawebthree.product.dto.ProductDetailDTO;
import com.metawebthree.product.infrastructure.persistence.mapper.ProductMapper;
import com.metawebthree.product.infrastructure.persistence.mapper.ProductEntityMapper;
import com.metawebthree.product.domain.model.ProductDO;
import com.metawebthree.product.domain.model.ProductEntityDO;

import lombok.extern.slf4j.Slf4j;

@Slf4j
@DubboService
public class ProductServiceRpcImpl implements com.metawebthree.common.generated.rpc.ProductService {

    private final ProductService productService;
    private final ProductMapper productMapper;
    private final ProductEntityMapper productEntityMapper;

    public ProductServiceRpcImpl(ProductService productService, ProductMapper productMapper, ProductEntityMapper productEntityMapper) {
        this.productService = productService;
        this.productMapper = productMapper;
        this.productEntityMapper = productEntityMapper;
    }

    @Override
    public GetProductDetailResponse getProductDetail(GetProductDetailRequest request) {
        ProductDetailDTO detail = productService.getProductDetail(ValidationUtils.safeIntFromLong(request.getProductId(), "productId"));
        if (detail == null) {
            return GetProductDetailResponse.newBuilder().build();
        }
        ProductDetailProto product = buildProductDetailProto(detail);
        return GetProductDetailResponse.newBuilder().setProduct(product).build();
    }

    private ProductDetailProto buildProductDetailProto(ProductDetailDTO detail) {
        ProductDetailProto.Builder builder = ProductDetailProto.newBuilder()
                .setId(detail.getId() != null ? detail.getId() : 0)
                .setName(detail.getGoodsName() != null ? detail.getGoodsName() : "")
                .setPic(detail.getImageUrl() != null ? detail.getImageUrl() : "")
                .setSubTitle(detail.getGoodsRemark() != null ? detail.getGoodsRemark() : "")
                .setPrice(detail.getSalePrice() != null ? detail.getSalePrice().doubleValue() : 0.0);
        if (detail.getPictures() != null) {
            builder.addAllPictures(detail.getPictures());
        }
        return builder.build();
    }

    @Override
    public CompletableFuture<GetProductDetailResponse> getProductDetailAsync(GetProductDetailRequest request) {
        return CompletableFuture.completedFuture(getProductDetail(request));
    }

    @Override
    public ListProductsResponse listProducts(ListProductsRequest request) {
        com.baomidou.mybatisplus.core.metadata.IPage<ProductDO> pageResult = queryListProducts(request);
        List<ProductDetailProto> products = toProductDetailProtos(pageResult);
        return ListProductsResponse.newBuilder()
                .addAllProducts(products)
                .setTotalCount((int) pageResult.getTotal())
                .setPage(request.getPage() > 0 ? request.getPage() : PaginationConstants.DEFAULT_PAGE)
                .setSize(request.getSize() > 0 ? request.getSize() : PaginationConstants.DEFAULT_SIZE)
                .build();
    }

    @Override
    public CompletableFuture<ListProductsResponse> listProductsAsync(ListProductsRequest request) {
        return CompletableFuture.completedFuture(listProducts(request));
    }

    @Override
    public GetProductBySkuResponse getProductBySku(GetProductBySkuRequest request) {
        return buildGetProductBySkuResponse(request);
    }

    @Override
    public CompletableFuture<GetProductBySkuResponse> getProductBySkuAsync(GetProductBySkuRequest request) {
        return CompletableFuture.completedFuture(getProductBySku(request));
    }

    @Override
    public CreateProductResponse createProduct(CreateProductRequest request) {
        ProductDO product = saveProduct(request);
        return CreateProductResponse.newBuilder()
                .setId(product.getId() != null ? product.getId() : 0L)
                .setSuccess(true)
                .setMessage("")
                .build();
    }

    @Override
    public CompletableFuture<CreateProductResponse> createProductAsync(CreateProductRequest request) {
        return CompletableFuture.completedFuture(createProduct(request));
    }

    @Override
    public UpdateProductResponse updateProduct(UpdateProductRequest request) {
        ProductDO product = productMapper.selectById(ValidationUtils.safeIntFromLong(request.getId(), "id"));
        if (product == null) {
            return UpdateProductResponse.newBuilder()
                    .setSuccess(false)
                    .setMessage("Product not found")
                    .build();
        }
        applyProductUpdate(product, request);
        applyEntityUpdate(request);
        return UpdateProductResponse.newBuilder()
                .setSuccess(true)
                .setMessage("Product updated successfully")
                .build();
    }

    @Override
    public CompletableFuture<UpdateProductResponse> updateProductAsync(UpdateProductRequest request) {
        return CompletableFuture.completedFuture(updateProduct(request));
    }

    @Override
    public DeleteProductResponse deleteProduct(DeleteProductRequest request) {
        int deleted = productMapper.deleteById(ValidationUtils.safeIntFromLong(request.getId(), "id"));
        return DeleteProductResponse.newBuilder()
                .setSuccess(deleted > 0)
                .setMessage(deleted > 0 ? "Product deleted successfully" : "Product not found")
                .build();
    }

    @Override
    public CompletableFuture<DeleteProductResponse> deleteProductAsync(DeleteProductRequest request) {
        return CompletableFuture.completedFuture(deleteProduct(request));
    }

    private com.baomidou.mybatisplus.core.metadata.IPage<ProductDO> queryListProducts(ListProductsRequest request) {
        int page = request.getPage() > 0 ? request.getPage() : PaginationConstants.DEFAULT_PAGE;
        int size = request.getSize() > 0 ? request.getSize() : PaginationConstants.DEFAULT_SIZE;

        com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper<ProductDO> wrapper =
                new com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper<ProductDO>()
                        .orderByDesc(ProductDO::getCreateTime);

        if (request.getCategoryId() != 0L) {
            wrapper.eq(ProductDO::getCategoryId, request.getCategoryId());
        }

        return productMapper.selectPage(
                new com.baomidou.mybatisplus.extension.plugins.pagination.Page<>(page, size),
                wrapper);
    }

    private List<ProductDetailProto> toProductDetailProtos(com.baomidou.mybatisplus.core.metadata.IPage<ProductDO> pageResult) {
        return pageResult.getRecords().stream()
                .map(p -> ProductDetailProto.newBuilder()
                        .setId(p.getId() != null ? p.getId() : 0)
                        .setName(p.getProductName() != null ? p.getProductName() : "")
                        .setSubTitle(p.getProductRemark() != null ? p.getProductRemark() : "")
                        .build())
                .toList();
    }

    private GetProductBySkuResponse buildGetProductBySkuResponse(GetProductBySkuRequest request) {
        List<ProductEntityDO> entities = productEntityMapper.selectList(
                new com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper<ProductEntityDO>()
                        .eq(ProductEntityDO::getSku, request.getSku()));
        if (entities.isEmpty()) return GetProductBySkuResponse.newBuilder().build();
        ProductEntityDO entity = entities.get(0);
        ProductDO product = productMapper.selectById(entity.getProductId());
        if (product == null) return GetProductBySkuResponse.newBuilder().build();
        ProductDetailProto proto = ProductDetailProto.newBuilder()
                .setId(product.getId() != null ? product.getId() : 0)
                .setName(product.getProductName() != null ? product.getProductName() : "")
                .setPrice(entity.getSalePrice() != null ? entity.getSalePrice().doubleValue() : 0.0)
                .setPic(entity.getImageUrl() != null ? entity.getImageUrl() : "")
                .build();
        return GetProductBySkuResponse.newBuilder().setProduct(proto).build();
    }

    private ProductDO saveProduct(CreateProductRequest request) {
        ProductDO product = new ProductDO();
        product.setProductName(request.getName());
        product.setSku(request.getSku());
        product.setProductRemark(request.getSubTitle());
        product.setCategoryId(request.getCategoryId());
        product.setCreateTime(java.time.LocalDateTime.now());
        productMapper.insert(product);

        ProductEntityDO entity = new ProductEntityDO();
        entity.setProductId(product.getId());
        entity.setSku(request.getSku());
        entity.setSalePrice(java.math.BigDecimal.valueOf(request.getPrice()));
        entity.setInventory(request.getStock());
        entity.setImageUrl(request.getPic());
        productEntityMapper.insert(entity);

        return product;
    }

    private void applyProductUpdate(ProductDO product, UpdateProductRequest request) {
        if (!request.getName().isEmpty()) {
            product.setProductName(request.getName());
        }
        if (!request.getSku().isEmpty()) {
            product.setSku(request.getSku());
        }
        if (!request.getSubTitle().isEmpty()) {
            product.setProductRemark(request.getSubTitle());
        }
        if (request.getCategoryId() != 0L) {
            product.setCategoryId(request.getCategoryId());
        }
        productMapper.updateById(product);
    }

    private void applyEntityUpdate(UpdateProductRequest request) {
        List<ProductEntityDO> entities = productEntityMapper.selectList(
                new com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper<ProductEntityDO>()
                        .eq(ProductEntityDO::getProductId, ValidationUtils.safeIntFromLong(request.getId(), "id")));
        if (entities.isEmpty()) return;
        ProductEntityDO entity = entities.get(0);
        if (!request.getSku().isEmpty()) entity.setSku(request.getSku());
        if (request.getPrice() != 0.0) entity.setSalePrice(java.math.BigDecimal.valueOf(request.getPrice()));
        if (request.getStock() != 0) entity.setInventory(request.getStock());
        if (!request.getPic().isEmpty()) entity.setImageUrl(request.getPic());
        productEntityMapper.updateById(entity);
    }

}
