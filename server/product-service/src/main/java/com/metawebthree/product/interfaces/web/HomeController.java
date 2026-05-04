package com.metawebthree.product.interfaces.web;

import com.metawebthree.common.dto.ApiResponse;
import com.metawebthree.product.dto.HomeContentDTO;
import com.metawebthree.product.dto.ProductDTO;
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
@RequestMapping("/home")
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

    @Operation(summary = "获取首页商品分类")
    @GetMapping("/productCateList")
    public ApiResponse<List<HomeContentDTO.CategoryDTO>> productCateList(
            @RequestParam(defaultValue = "0") Long parentId) {
        // TODO: 实现分类树形结构
        return ApiResponse.success(Collections.emptyList());
    }

    @Operation(summary = "分页获取人气推荐商品")
    @GetMapping("/hotProductList")
    public ApiResponse<List<ProductDTO>> hotProductList(
            @RequestParam(defaultValue = "1") Integer pageNum,
            @RequestParam(defaultValue = "10") Integer pageSize) {
        return ApiResponse.success(productService.listProducts(null, "hot", null));
    }

    @Operation(summary = "分页获取新品推荐商品")
    @GetMapping("/newProductList")
    public ApiResponse<List<ProductDTO>> newProductList(
            @RequestParam(defaultValue = "1") Integer pageNum,
            @RequestParam(defaultValue = "10") Integer pageSize) {
        return ApiResponse.success(productService.listProducts(null, "new", null));
    }

    @Operation(summary = "根据分类分页获取专题")
    @GetMapping("/subjectList")
    public ApiResponse<List<HomeContentDTO.SubjectDTO>> subjectList(
            @RequestParam(required = false) Long categoryId,
            @RequestParam(defaultValue = "1") Integer pageNum,
            @RequestParam(defaultValue = "10") Integer pageSize) {
        // TODO: 接入专题服务后返回真实数据
        return ApiResponse.success(Collections.emptyList());
    }
}
