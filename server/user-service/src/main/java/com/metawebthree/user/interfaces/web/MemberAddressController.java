package com.metawebthree.user.interfaces.web;

import com.metawebthree.common.constants.HeaderConstants;
import com.metawebthree.common.constants.RequestHeaderKeys;
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
@RequestMapping("/member/address")
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
    @GetMapping
    public ApiResponse<List<MemberAddress>> list(@RequestHeader(HeaderConstants.USER_ID) Long memberId) {
        return ApiResponse.success(addressService.listAddresses(memberId));
    }

    @Operation(summary = "Get address by ID")
    @GetMapping("/{id}")
    public ApiResponse<MemberAddress> getById(@PathVariable Long id) {
        return ApiResponse.success(addressService.getAddressById(id));
    }

    @Operation(summary = "Update address")
    @PostMapping("/update/{id}")
    public ApiResponse<Void> update(@PathVariable Long id, @RequestBody MemberAddress address) {
        address.setId(id);
        addressService.updateAddress(address);
        return ApiResponse.success();
    }

    @Operation(summary = "Remove address")
    @PostMapping("/delete/{id}")
    public ApiResponse<Void> remove(@PathVariable Long id) {
        addressService.removeAddress(id);
        return ApiResponse.success();
    }
}
