package com.metawebthree.product.application;

import java.util.List;
import java.util.concurrent.CompletableFuture;

import org.apache.dubbo.config.annotation.DubboService;
import org.springframework.beans.factory.annotation.Autowired;

import com.metawebthree.common.generated.rpc.GetProductDetailRequest;
import com.metawebthree.common.generated.rpc.GetProductDetailResponse;
import com.metawebthree.common.generated.rpc.ProductDetailProto;
import com.metawebthree.common.generated.rpc.ProductService;
import com.metawebthree.product.dto.ProductDetailDTO;

import lombok.extern.slf4j.Slf4j;

@Slf4j
@DubboService
public class ProductServiceRpcImpl implements ProductService {

    @Autowired
    private ProductService productService;

    @Override
    public GetProductDetailResponse getProductDetail(GetProductDetailRequest request) {
        log.info("Dubbo RPC: getProductDetail called with productId: {}", request.getProductId());
        
        ProductDetailDTO detail = productService.getProductDetail(request.getProductId().intValue());
        
        if (detail == null) {
            return GetProductDetailResponse.newBuilder().build();
        }

        ProductDetailProto.Builder builder = ProductDetailProto.newBuilder()
                .setId(detail.getId() != null ? detail.getId() : 0)
                .setName(detail.getGoodsName() != null ? detail.getGoodsName() : "")
                .setPic(detail.getImageUrl() != null ? detail.getImageUrl() : "")
                .setSubTitle(detail.getGoodsRemark() != null ? detail.getGoodsRemark() : "")
                .setPrice(detail.getSalePrice() != null ? detail.getSalePrice().doubleValue() : 0.0);

        if (detail.getPictures() != null) {
            builder.addAllPictures(detail.getPictures());
        }

        return GetProductDetailResponse.newBuilder()
                .setProduct(builder.build())
                .build();
    }

    @Override
    public CompletableFuture<GetProductDetailResponse> getProductDetailAsync(GetProductDetailRequest request) {
        return CompletableFuture.completedFuture(getProductDetail(request));
    }
}