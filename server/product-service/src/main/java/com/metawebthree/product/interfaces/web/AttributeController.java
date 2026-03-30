package com.metawebthree.product.interfaces.web;

import com.metawebthree.common.dto.ApiResponse;
import com.metawebthree.product.application.AttributeApplicationService;
import com.metawebthree.product.domain.model.Attribute;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;
import lombok.RequiredArgsConstructor;
import org.springframework.validation.annotation.Validated;
import org.springframework.web.bind.annotation.*;
import java.util.List;

@Validated
@RestController
@RequestMapping("/v1/attributes")
@RequiredArgsConstructor
@Tag(name = "Product Attribute Management")
public class AttributeController {

    private final AttributeApplicationService attributeService;

    @Operation(summary = "Define new attribute")
    @PostMapping
    public ApiResponse<Void> define(@RequestBody Attribute attribute) {
        attributeService.defineAttribute(attribute);
        return ApiResponse.success();
    }

    @Operation(summary = "Get attribute by ID")
    @GetMapping("/{id}")
    public ApiResponse<Attribute> details(@PathVariable Long id) {
        return ApiResponse.success(attributeService.getAttribute(id));
    }

    @Operation(summary = "Find attributes of category")
    @GetMapping("/category/{categoryId}")
    public ApiResponse<List<Attribute>> listByCategory(@PathVariable Long categoryId) {
        return ApiResponse.success(attributeService.findByCategory(categoryId));
    }

    @Operation(summary = "Modify existing attribute")
    @PutMapping("/{id}")
    public ApiResponse<Void> modify(@PathVariable Long id, @RequestBody Attribute attribute) {
        attributeService.modifyAttribute(attribute);
        return ApiResponse.success();
    }

    @Operation(summary = "Delete attribute")
    @DeleteMapping("/{id}")
    public ApiResponse<Void> remove(@PathVariable Long id) {
        attributeService.removeAttribute(id);
        return ApiResponse.success();
    }
}
