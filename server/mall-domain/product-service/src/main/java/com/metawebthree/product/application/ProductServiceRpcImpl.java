package com.metawebthree.product.application;

import java.util.List;
import java.util.concurrent.CompletableFuture;

import org.apache.dubbo.config.annotation.DubboService;
import org.springframework.beans.factory.annotation.Autowired;

import com.metawebthree.common.generated.rpc.*;
import com.metawebthree.product.dto.ProductDetailDTO;
import com.metawebthree.product.infrastructure.persistence.mapper.ProductMapper;
import com.metawebthree.product.infrastructure.persistence.mapper.ProductEntityMapper;
import com.metawebthree.product.domain.model.ProductDO;
import com.metawebthree.product.domain.model.ProductEntityDO;

import lombok.extern.slf4j.Slf4j;

@Slf4j
@DubboService
public class ProductServiceRpcImpl implements com.metawebthree.common.generated.rpc.ProductService {

    @Autowired
    private ProductService productService;

    @Autowired
    private ProductMapper productMapper;

    @Autowired
    private ProductEntityMapper productEntityMapper;

    @Override
    public GetProductDetailResponse getProductDetail(GetProductDetailRequest request) {
        ProductDetailDTO detail = productService.getProductDetail((int) request.getProductId());
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
        log.info("Dubbo RPC: listProducts called with page: {}, size: {}, categoryId: {}",
                request.getPage(), request.getSize(), request.getCategoryId());
        try {
            com.baomidou.mybatisplus.core.metadata.IPage<ProductDO> pageResult = queryListProducts(request);
            List<ProductDetailProto> products = toProductDetailProtos(pageResult);
            return ListProductsResponse.newBuilder()
                    .addAllProducts(products)
                    .setTotalCount((int) pageResult.getTotal())
                    .setPage(request.getPage() > 0 ? request.getPage() : 1)
                    .setSize(request.getSize() > 0 ? request.getSize() : 10)
                    .build();
        } catch (Exception e) {
            log.error("Failed to list products", e);
            return ListProductsResponse.newBuilder().build();
        }
    }

    @Override
    public CompletableFuture<ListProductsResponse> listProductsAsync(ListProductsRequest request) {
        return CompletableFuture.completedFuture(listProducts(request));
    }

    @Override
    public GetProductBySkuResponse getProductBySku(GetProductBySkuRequest request) {
        log.info("Dubbo RPC: getProductBySku called with sku: {}", request.getSku());
        try {
            return buildGetProductBySkuResponse(request);
        } catch (Exception e) {
            log.error("Failed to get product by sku", e);
            return GetProductBySkuResponse.newBuilder().build();
        }
    }

    @Override
    public CompletableFuture<GetProductBySkuResponse> getProductBySkuAsync(GetProductBySkuRequest request) {
        return CompletableFuture.completedFuture(getProductBySku(request));
    }

    @Override
    public CreateProductResponse createProduct(CreateProductRequest request) {
        log.info("Dubbo RPC: createProduct called with name: {}, sku: {}", request.getName(), request.getSku());
        try {
            ProductDO product = saveProduct(request);
            return CreateProductResponse.newBuilder()
                    .setId(product.getId() != null ? product.getId() : 0L)
                    .setSuccess(true)
                    .setMessage("Product created successfully")
                    .build();
        } catch (Exception e) {
            log.error("Failed to create product", e);
            return CreateProductResponse.newBuilder()
                    .setSuccess(false)
                    .setMessage("Failed to create product: " + e.getMessage())
                    .build();
        }
    }

    @Override
    public CompletableFuture<CreateProductResponse> createProductAsync(CreateProductRequest request) {
        return CompletableFuture.completedFuture(createProduct(request));
    }

    @Override
    public UpdateProductResponse updateProduct(UpdateProductRequest request) {
        log.info("Dubbo RPC: updateProduct called with id: {}", request.getId());
        try {
            ProductDO product = productMapper.selectById((int) request.getId());
            if (product == null) return buildUpdateErrorResponse("Product not found");
            applyProductUpdate(product, request);
            applyEntityUpdate(request);
            return buildUpdateSuccessResponse();
        } catch (Exception e) {
            log.error("Failed to update product", e);
            return buildUpdateErrorResponse("Failed to update product: " + e.getMessage());
        }
    }

    @Override
    public CompletableFuture<UpdateProductResponse> updateProductAsync(UpdateProductRequest request) {
        return CompletableFuture.completedFuture(updateProduct(request));
    }

    @Override
    public DeleteProductResponse deleteProduct(DeleteProductRequest request) {
        log.info("Dubbo RPC: deleteProduct called with id: {}", request.getId());
        try {
            int deleted = productMapper.deleteById((int) request.getId());
            return DeleteProductResponse.newBuilder()
                    .setSuccess(deleted > 0)
                    .setMessage(deleted > 0 ? "Product deleted successfully" : "Product not found")
                    .build();
        } catch (Exception e) {
            log.error("Failed to delete product", e);
            return DeleteProductResponse.newBuilder()
                    .setSuccess(false)
                    .setMessage("Failed to delete product: " + e.getMessage())
                    .build();
        }
    }

    @Override
    public CompletableFuture<DeleteProductResponse> deleteProductAsync(DeleteProductRequest request) {
        return CompletableFuture.completedFuture(deleteProduct(request));
    }

    private com.baomidou.mybatisplus.core.metadata.IPage<ProductDO> queryListProducts(ListProductsRequest request) {
        int page = request.getPage() > 0 ? request.getPage() : 1;
        int size = request.getSize() > 0 ? request.getSize() : 10;

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
                        .eq(ProductEntityDO::getProductArtno, request.getSku()));
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
        product.setProductNo(request.getSku());
        product.setProductRemark(request.getSubTitle());
        product.setCategoryId(request.getCategoryId());
        product.setCreateTime(java.time.LocalDateTime.now());
        productMapper.insert(product);

        ProductEntityDO entity = new ProductEntityDO();
        entity.setProductId(product.getId());
        entity.setProductArtno(request.getSku());
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
            product.setProductNo(request.getSku());
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
        if (request.getSku().isEmpty() && request.getPrice() == 0.0 && request.getStock() == 0 && request.getPic().isEmpty()) return;
        List<ProductEntityDO> entities = productEntityMapper.selectList(
                new com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper<ProductEntityDO>()
                        .eq(ProductEntityDO::getProductId, (int) request.getId()));
        if (entities.isEmpty()) return;
        ProductEntityDO entity = entities.get(0);
        if (!request.getSku().isEmpty()) entity.setProductArtno(request.getSku());
        if (request.getPrice() != 0.0) entity.setSalePrice(java.math.BigDecimal.valueOf(request.getPrice()));
        if (request.getStock() != 0) entity.setInventory(request.getStock());
        if (!request.getPic().isEmpty()) entity.setImageUrl(request.getPic());
        productEntityMapper.updateById(entity);
    }

    private UpdateProductResponse buildUpdateErrorResponse(String message) {
        return UpdateProductResponse.newBuilder()
                .setSuccess(false)
                .setMessage(message)
                .build();
    }

    private UpdateProductResponse buildUpdateSuccessResponse() {
        return UpdateProductResponse.newBuilder()
                .setSuccess(true)
                .setMessage("Product updated successfully")
                .build();
    }
}
