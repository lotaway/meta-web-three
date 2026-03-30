package com.metawebthree.user.interfaces.web;

import com.metawebthree.common.dto.ApiResponse;
import com.metawebthree.user.application.MemberAddressApplicationService;
import com.metawebthree.user.domain.model.MemberAddress;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;
import lombok.RequiredArgsConstructor;
import org.springframework.validation.annotation.Validated;
import org.springframework.web.bind.annotation.*;
import java.util.List;

@Validated
@RestController
@RequestMapping("/v1/addresses")
@RequiredArgsConstructor
@Tag(name = "Member Address Management")
public class MemberAddressController {

    private final MemberAddressApplicationService addressService;

    @Operation(summary = "Add address")
    @PostMapping
    public ApiResponse<Void> add(@RequestBody MemberAddress address) {
        addressService.addAddress(address);
        return ApiResponse.success();
    }

    @Operation(summary = "List all member addresses")
    @GetMapping("/member/{memberId}")
    public ApiResponse<List<MemberAddress>> list(@PathVariable Long memberId) {
        return ApiResponse.success(addressService.listAddresses(memberId));
    }

    @Operation(summary = "Update address")
    @PutMapping("/{id}")
    public ApiResponse<Void> update(@PathVariable Long id, @RequestBody MemberAddress address) {
        addressService.updateAddress(address);
        return ApiResponse.success();
    }

    @Operation(summary = "Remove address")
    @DeleteMapping("/{id}")
    public ApiResponse<Void> remove(@PathVariable Long id) {
        addressService.removeAddress(id);
        return ApiResponse.success();
    }
}
