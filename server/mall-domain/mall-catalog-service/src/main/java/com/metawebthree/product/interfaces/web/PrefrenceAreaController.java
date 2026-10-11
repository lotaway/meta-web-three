package com.metawebthree.product.interfaces.web;

import com.metawebthree.common.dto.ApiResponse;
import com.metawebthree.product.domain.model.PrefrenceAreaDO;
import com.metawebthree.product.infrastructure.persistence.mapper.PrefrenceAreaMapper;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;
import lombok.RequiredArgsConstructor;
import org.springframework.validation.annotation.Validated;
import org.springframework.web.bind.annotation.*;
import java.util.List;

@Validated
@RestController
@RequestMapping("/prefrenceArea")
@RequiredArgsConstructor
@Tag(name = "Prefrence Area Management")
public class PrefrenceAreaController {

    private final PrefrenceAreaMapper prefrenceAreaMapper;

    @Operation(summary = "获取所有优选专区")
    @GetMapping("/listAll")
    public ApiResponse<List<PrefrenceAreaDO>> listAll() {
        return ApiResponse.success(prefrenceAreaMapper.selectList(null));
    }
}
