package com.metawebthree.promotion.interfaces.web;

import com.metawebthree.common.dto.ApiResponse;
import com.metawebthree.promotion.application.service.CmsPrefrenceAreaService;
import com.metawebthree.promotion.infrastructure.persistence.model.CmsPrefrenceAreaDO;
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
@Tag(name = "CMS Prefrence Area Admin")
public class CmsPrefrenceAreaController {
    private final CmsPrefrenceAreaService prefrenceAreaService;

    @Operation(summary = "获取所有优选专区")
    @GetMapping("/listAll")
    public ApiResponse<List<CmsPrefrenceAreaDO>> listAll() {
        return ApiResponse.success(prefrenceAreaService.listAll());
    }
}
