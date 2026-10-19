package com.metawebthree.product.interfaces.web;

import com.metawebthree.common.dto.ApiResponse;
import com.metawebthree.product.application.HomeContentService;
import com.metawebthree.product.dto.HomeContentDTO;
import com.metawebthree.product.dto.ProductDTO;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@Controller
@RequestMapping("/home")
@RequiredArgsConstructor
@Tag(name = "Home Controller", description = "首页内容接口")
public class HomeController {

    private final HomeContentService homeContentService;

    @Operation(summary = "首页内容展示")
    @GetMapping("/content")
    @ResponseBody
    public ApiResponse<HomeContentDTO> content() {
        return ApiResponse.success(homeContentService.content());
    }

    @Operation(summary = "分页获取推荐商品")
    @GetMapping("/recommendProductList")
    @ResponseBody
    public ApiResponse<List<ProductDTO>> recommendProductList(
            @RequestParam(defaultValue = "1") Integer pageNum,
            @RequestParam(defaultValue = "10") Integer pageSize) {
        return ApiResponse.success(homeContentService.recommendProductList(pageNum, pageSize));
    }

    @Operation(summary = "获取首页商品分类")
    @GetMapping("/productCateList")
    @ResponseBody
    public ApiResponse<List<HomeContentDTO.CategoryDTO>> productCateList(
            @RequestParam(defaultValue = "0") Long parentId) {
        return ApiResponse.success(homeContentService.productCateList(parentId));
    }

    @Operation(summary = "分页获取人气推荐商品")
    @GetMapping("/hotProductList")
    @ResponseBody
    public ApiResponse<List<ProductDTO>> hotProductList(
            @RequestParam(defaultValue = "1") Integer pageNum,
            @RequestParam(defaultValue = "10") Integer pageSize) {
        return ApiResponse.success(homeContentService.hotProductList(pageNum, pageSize));
    }

    @Operation(summary = "分页获取新品推荐商品")
    @GetMapping("/newProductList")
    @ResponseBody
    public ApiResponse<List<ProductDTO>> newProductList(
            @RequestParam(defaultValue = "1") Integer pageNum,
            @RequestParam(defaultValue = "10") Integer pageSize) {
        return ApiResponse.success(homeContentService.newProductList(pageNum, pageSize));
    }

    @Operation(summary = "根据分类分页获取专题")
    @GetMapping("/subjectList")
    @ResponseBody
    public ApiResponse<List<HomeContentDTO.SubjectDTO>> subjectList(
            @RequestParam(required = false) Long categoryId,
            @RequestParam(defaultValue = "1") Integer pageNum,
            @RequestParam(defaultValue = "10") Integer pageSize) {
        return ApiResponse.success(homeContentService.subjectList(categoryId, pageNum, pageSize));
    }
}
