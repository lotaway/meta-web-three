package com.metawebthree.product.interfaces.web;

import com.metawebthree.product.application.ProductService;
import com.metawebthree.product.domain.exception.ProductDomainException;
import com.metawebthree.product.domain.exception.ProductErrorCode;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.Parameter;
import io.swagger.v3.oas.annotations.tags.Tag;
import jakarta.validation.constraints.NotNull;
import org.springframework.http.MediaType;
import org.springframework.validation.annotation.Validated;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.core.conditions.update.UpdateWrapper;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.metawebthree.common.dto.ApiResponse;
import com.metawebthree.product.dto.ProductDTO;
import com.metawebthree.product.dto.ProductDetailDTO;
import com.metawebthree.product.application.ProductCategoryApplicationService;
import com.metawebthree.product.domain.model.AdminProductDO;
import com.metawebthree.product.domain.model.ProductCategory;
import com.metawebthree.product.infrastructure.persistence.mapper.AdminProductMapper;
import com.metawebthree.product.infrastructure.persistence.mapper.ProductMapper;
import com.metawebthree.product.interfaces.web.dto.ProductCategoryNode;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

@Validated
@RestController
@RequestMapping("/product")
@Tag(name = "Product Management", description = "Endpoints for product lifecycle and management")
public class ProductController {

    private final ProductService productService;
    private final ProductCategoryApplicationService categoryService;
    private final AdminProductMapper adminProductMapper;
    private final ProductMapper productMapper;

    public ProductController(ProductService productService, ProductCategoryApplicationService categoryService,
                             AdminProductMapper adminProductMapper, ProductMapper productMapper) {
        this.productService = productService;
        this.categoryService = categoryService;
        this.adminProductMapper = adminProductMapper;
        this.productMapper = productMapper;
    }

    @Operation(summary = "Check service status", description = "Basic health check for product service")
    @GetMapping("/health")
    public ApiResponse<String> healthCheck() {
        return ApiResponse.success("product-service is running");
    }

    @Operation(summary = "Create a new product")
    @PostMapping
    public ApiResponse<Void> createProduct() {
        productService.createProduct();
        return ApiResponse.success();
    }

    @Operation(summary = "List products with filters")
    @GetMapping
    public ApiResponse<List<ProductDTO>> listProducts(
            @Parameter(description = "Category ID") @RequestParam(required = false) Integer categoryId,
            @Parameter(description = "Search keyword") @RequestParam(required = false) String keyword,
            @Parameter(description = "Price range") @RequestParam(required = false) String priceRange) {
        return ApiResponse.success(productService.listProducts(categoryId, keyword, priceRange));
    }

    @Operation(summary = "Get detailed product info by ID (alias /detail/{id})")
    @GetMapping({"/{id}", "/detail/{id}"})
    public ApiResponse<ProductDetailDTO> getProduct(
            @Parameter(description = "Product ID") @PathVariable @NotNull Integer id) {
        ProductDetailDTO detail = productService.getProductDetail(id);
        if (detail == null) {
            throw new ProductDomainException(ProductErrorCode.NOT_FOUND, "Product not found with id: " + id);
        }
        return ApiResponse.success(detail);
    }

    @Operation(summary = "Update product content", description = "Updates the description/content of an existing product")
    @PutMapping("/{id}")
    public ApiResponse<Void> updateProduct(
            @Parameter(description = "Product ID") @PathVariable @NotNull Integer id,
            @RequestBody String content) {
        productService.updateProduct(Long.valueOf(id), content.getBytes());
        return ApiResponse.success();
    }

    @Operation(summary = "Delete product", description = "Deletes a product by its ID")
    @DeleteMapping("/{id}")
    public ApiResponse<Void> deleteProduct(
            @Parameter(description = "Product ID") @PathVariable @NotNull Integer id) {
        productService.deleteProduct(id.toString());
        return ApiResponse.success();
    }

    @Operation(summary = "Upload product image", description = "Uploads an image for a specific product")
    @PostMapping(path = "/{id}/images", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
    public ApiResponse<Void> uploadImage(
            @Parameter(description = "Product ID") @PathVariable Integer id,
            @Parameter(description = "Image file") @RequestParam MultipartFile file) {
        productService.uploadImage(Long.valueOf(id), file);
        return ApiResponse.success();
    }

    @Operation(summary = "Search products", description = "Comprehensive product search with filters and sorting")
    @GetMapping("/search")
    public ApiResponse<List<ProductDTO>> searchProducts(
            @Parameter(description = "Search keyword") @RequestParam(required = false) String keyword,
            @Parameter(description = "Category ID") @RequestParam(required = false) Integer categoryId,
            @Parameter(description = "Brand ID") @RequestParam(required = false) Integer brandId,
            @Parameter(description = "Sort type: 0-综合, 1-销量, 2-价格升序, 3-价格降序") @RequestParam(defaultValue = "0") Integer sort,
            @Parameter(description = "Page number") @RequestParam(defaultValue = "1") Integer pageNum,
            @Parameter(description = "Page size") @RequestParam(defaultValue = "20") Integer pageSize) {
        return ApiResponse.success(productService.searchProducts(keyword, categoryId, brandId, sort, pageNum, pageSize));
    }

    @Operation(summary = "Simple search", description = "Simple product search by keyword")
    @GetMapping("/search/simple")
    public ApiResponse<List<ProductDTO>> simpleSearch(
            @Parameter(description = "Search keyword") @RequestParam String keyword,
            @Parameter(description = "Limit") @RequestParam(defaultValue = "10") Integer limit) {
        return ApiResponse.success(productService.simpleSearch(keyword, limit));
    }

    @Operation(summary = "Recommend products", description = "Recommend products based on product ID")
    @GetMapping("/recommend/{id}")
    public ApiResponse<List<ProductDTO>> recommendProducts(
            @Parameter(description = "Product ID") @PathVariable Integer id,
            @Parameter(description = "Limit") @RequestParam(defaultValue = "5") Integer limit) {
        return ApiResponse.success(productService.recommendProducts(id, limit));
    }

    @Operation(summary = "Get category tree list (alias)")
    @GetMapping("/categoryTreeList")
    public ApiResponse<List<ProductCategoryNode>> categoryTreeList() {
        return ApiResponse.success(categoryService.categoryTreeList());
    }

    // === Admin endpoints ===

    @Operation(summary = "商品简单列表(用于选择)")
    @GetMapping("/simpleList")
    public ApiResponse<List<AdminProductDO>> simpleList() {
        LambdaQueryWrapper<AdminProductDO> wrapper = new LambdaQueryWrapper<AdminProductDO>()
                .select(AdminProductDO::getId, AdminProductDO::getName);
        return ApiResponse.success(adminProductMapper.selectList(wrapper));
    }

    @Operation(summary = "分页查询商品列表")
    @GetMapping("/list")
    public ApiResponse<Page<AdminProductDO>> list(
            @RequestParam(defaultValue = "1") Integer pageNum,
            @RequestParam(defaultValue = "10") Integer pageSize,
            @RequestParam(required = false) String keyword,
            @RequestParam(required = false) Long brandId,
            @RequestParam(required = false) Long productCategoryId,
            @RequestParam(required = false) Integer publishStatus,
            @RequestParam(required = false) Integer verifyStatus) {
        LambdaQueryWrapper<AdminProductDO> wrapper = new LambdaQueryWrapper<>();
        if (keyword != null && !keyword.isEmpty()) {
            wrapper.like(AdminProductDO::getName, keyword);
        }
        if (brandId != null) {
            wrapper.eq(AdminProductDO::getBrandId, brandId);
        }
        if (productCategoryId != null) {
            wrapper.eq(AdminProductDO::getProductCategoryId, productCategoryId);
        }
        if (publishStatus != null) {
            wrapper.eq(AdminProductDO::getPublishStatus, publishStatus);
        }
        if (verifyStatus != null) {
            wrapper.eq(AdminProductDO::getVerifyStatus, verifyStatus);
        }
        wrapper.orderByDesc(AdminProductDO::getId);
        return ApiResponse.success(adminProductMapper.selectPage(new Page<>(pageNum, pageSize), wrapper));
    }

    @Operation(summary = "批量修改删除状态")
    @PostMapping("/update/deleteStatus")
    public ApiResponse<Void> updateDeleteStatus(
            @RequestParam String ids,
            @RequestParam Integer deleteStatus) {
        List<Long> idList = Arrays.stream(ids.split(","))
                .map(String::trim).map(Long::parseLong).collect(Collectors.toList());
        adminProductMapper.update(null, new UpdateWrapper<AdminProductDO>()
                .in("id", idList).set("delete_status", deleteStatus));
        return ApiResponse.success();
    }

    @Operation(summary = "批量设为新品")
    @PostMapping("/update/newStatus")
    public ApiResponse<Void> updateNewStatus(
            @RequestParam String ids,
            @RequestParam Integer newStatus) {
        List<Long> idList = Arrays.stream(ids.split(","))
                .map(String::trim).map(Long::parseLong).collect(Collectors.toList());
        adminProductMapper.update(null, new UpdateWrapper<AdminProductDO>()
                .in("id", idList).set("new_status", newStatus));
        return ApiResponse.success();
    }

    @Operation(summary = "批量推荐商品")
    @PostMapping("/update/recommendStatus")
    public ApiResponse<Void> updateRecommendStatus(
            @RequestParam String ids,
            @RequestParam Integer recommendStatus) {
        List<Long> idList = Arrays.stream(ids.split(","))
                .map(String::trim).map(Long::parseLong).collect(Collectors.toList());
        adminProductMapper.update(null, new UpdateWrapper<AdminProductDO>()
                .in("id", idList).set("recommand_status", recommendStatus));
        return ApiResponse.success();
    }

    @Operation(summary = "批量上下架商品")
    @PostMapping("/update/publishStatus")
    public ApiResponse<Void> updatePublishStatus(
            @RequestParam String ids,
            @RequestParam Integer publishStatus) {
        List<Long> idList = Arrays.stream(ids.split(","))
                .map(String::trim).map(Long::parseLong).collect(Collectors.toList());
        adminProductMapper.update(null, new UpdateWrapper<AdminProductDO>()
                .in("id", idList).set("publish_status", publishStatus));
        return ApiResponse.success();
    }

    @Operation(summary = "批量审核商品")
    @PostMapping("/update/verifyStatus")
    public ApiResponse<Void> updateVerifyStatus(
            @RequestParam String ids,
            @RequestParam Integer verifyStatus) {
        List<Long> idList = Arrays.stream(ids.split(","))
                .map(String::trim).map(Long::parseLong).collect(Collectors.toList());
        adminProductMapper.update(null, new UpdateWrapper<AdminProductDO>()
                .in("id", idList).set("verify_status", verifyStatus));
        return ApiResponse.success();
    }

    @Operation(summary = "创建商品")
    @PostMapping("/create")
    public ApiResponse<Void> create(@RequestBody AdminProductDO product) {
        adminProductMapper.insert(product);
        return ApiResponse.success();
    }

    @Operation(summary = "根据ID修改商品信息")
    @PostMapping("/update/{id}")
    public ApiResponse<Void> updateById(@PathVariable Long id, @RequestBody AdminProductDO product) {
        product.setId(id);
        adminProductMapper.updateById(product);
        return ApiResponse.success();
    }

    @Operation(summary = "根据商品id获取商品编辑信息")
    @GetMapping("/updateInfo/{id}")
    public ApiResponse<AdminProductDO> updateInfo(@PathVariable Long id) {
        AdminProductDO product = adminProductMapper.selectById(id);
        return ApiResponse.success(product);
    }
}
