package com.metawebthree.product.interfaces.web;

import com.metawebthree.common.dto.ApiResponse;
import com.metawebthree.product.application.EsProductApplicationService;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.Parameter;
import io.swagger.v3.oas.annotations.tags.Tag;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/esProduct")
@Tag(name = "ES Product Management", description = "Elasticsearch product synchronization endpoints")
public class EsProductController {

    private final EsProductApplicationService esProductService;

    public EsProductController(EsProductApplicationService esProductService) {
        this.esProductService = esProductService;
    }

    @PostMapping("/importAll")
    @Operation(summary = "Import all products to Elasticsearch")
    public ApiResponse<Integer> importAll() {
        int count = esProductService.importAllToEs();
        return ApiResponse.success(count);
    }

    @PostMapping("/create/{id}")
    @Operation(summary = "Sync single product to Elasticsearch")
    public ApiResponse<Void> create(
            @Parameter(description = "Product ID") @PathVariable Integer id) {
        esProductService.syncToEs(id);
        return ApiResponse.success();
    }
}
