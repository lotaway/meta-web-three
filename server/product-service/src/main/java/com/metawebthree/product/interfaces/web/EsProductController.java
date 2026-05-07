package com.metawebthree.product.interfaces.web;

import com.metawebthree.common.dto.ApiResponse;
import com.metawebthree.product.application.EsProductApplicationService;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/esProduct")
public class EsProductController {

    private final EsProductApplicationService esProductService;

    public EsProductController(EsProductApplicationService esProductService) {
        this.esProductService = esProductService;
    }

    @PostMapping("/importAll")
    public ApiResponse<Integer> importAll() {
        int count = esProductService.importAllToEs();
        return ApiResponse.success(count);
    }

    @PostMapping("/create/{id}")
    public ApiResponse<Void> create(@PathVariable Integer id) {
        esProductService.syncToEs(id);
        return ApiResponse.success();
    }
}
