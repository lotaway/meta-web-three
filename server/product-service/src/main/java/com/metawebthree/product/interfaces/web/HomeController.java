package com.metawebthree.product.interfaces.web;

import com.metawebthree.common.dto.ApiResponse;
import com.metawebthree.common.dto.HomeContentDTO;
import com.metawebthree.common.dto.ProductDTO;
import com.metawebthree.product.application.BrandApplicationService;
import com.metawebthree.product.application.ProductService;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.client.RestTemplate;

import java.util.Collections;
import java.util.List;
import java.util.stream.Collectors;

@RestController
@RequestMapping("/v1/home")
@RequiredArgsConstructor
@Tag(name = "Home Controller", description = "首页内容接口")
public class HomeController {

    private final ProductService productService;
    private final BrandApplicationService brandService;
    private final RestTemplate restTemplate;

    @Operation(summary = "首页内容展示")
    @GetMapping("/content")
    public ApiResponse<HomeContentDTO> content() {
        List<ProductDTO> newProducts = productService.listProducts(null, "new", null);
        List<ProductDTO> hotProducts = productService.listProducts(null, "hot", null);
        List<HomeContentDTO.BrandDTO> brands = brandService.listBrands().stream()
                .map(b -> HomeContentDTO.BrandDTO.builder()
                        .id(b.getId())
                        .name(b.getName())
                        .logo(b.getLogo())
                        .productCount(b.getProductCount())
                        .build())
                .collect(Collectors.toList());

        // 通过 RestTemplate 调用 promotion-service 获取广告
        // 实际生产中建议使用 Dubbo 或 OpenFeign
        String advertiseUrl = "http://promotion-service/v1/promotion/advertises/available?type=1";
        List<HomeContentDTO.AdvertiseDTO> advertiseList = Collections.emptyList();
        try {
            ApiResponse<?> response = restTemplate.getForObject(advertiseUrl, ApiResponse.class);
            Object data = response != null ? response.getData() : null;
            if (data instanceof List<?> list) {
                advertiseList = list.stream().map(obj -> {
                    var map = (java.util.Map<?, ?>) obj;
                    Object id = map.get("id");
                    Object name = map.get("name");
                    Object pic = map.get("pic");
                    Object url = map.get("url");
                    return HomeContentDTO.AdvertiseDTO.builder()
                            .id(id == null ? null : Long.valueOf(String.valueOf(id)))
                            .name(name == null ? null : String.valueOf(name))
                            .pic(pic == null ? null : String.valueOf(pic))
                            .link(url == null ? null : String.valueOf(url))
                            .build();
                }).collect(Collectors.toList());
            } 
        } catch (Exception e) {
            advertiseList = Collections.emptyList();
        }

        HomeContentDTO content = HomeContentDTO.builder()
                .newProductList(newProducts)
                .hotProductList(hotProducts)
                .brandList(brands)
                .advertiseList(advertiseList)
                .subjectList(Collections.emptyList())   
                .homeFlashPromotion(null)               
                .build();

        return ApiResponse.success(content);
    }

    @Operation(summary = "分页获取推荐商品")
    @GetMapping("/recommendProductList")
    public ApiResponse<List<ProductDTO>> recommendProductList(
            @RequestParam(defaultValue = "1") Integer pageNum,
            @RequestParam(defaultValue = "10") Integer pageSize) {
        // 简单实现：按推荐关键字搜索
        return ApiResponse.success(productService.listProducts(null, "recommend", null));
    }
}
