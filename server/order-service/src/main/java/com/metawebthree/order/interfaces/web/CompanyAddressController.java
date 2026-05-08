package com.metawebthree.order.interfaces.web;

import com.metawebthree.common.dto.ApiResponse;
import com.metawebthree.order.domain.model.CompanyAddressDO;
import com.metawebthree.order.infrastructure.persistence.mapper.CompanyAddressMapper;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;
import lombok.RequiredArgsConstructor;
import org.springframework.validation.annotation.Validated;
import org.springframework.web.bind.annotation.*;
import java.util.List;

@Validated
@RestController
@RequestMapping("/companyAddress")
@RequiredArgsConstructor
@Tag(name = "Company Address Management")
public class CompanyAddressController {

    private final CompanyAddressMapper companyAddressMapper;

    @Operation(summary = "获取所有公司收货地址")
    @GetMapping("/list")
    public ApiResponse<List<CompanyAddressDO>> list() {
        return ApiResponse.success(companyAddressMapper.selectList(null));
    }
}
