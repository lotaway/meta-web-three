package com.metawebthree.wallet.interfaces.admin;

import com.metawebthree.common.dto.ApiResponse;
import com.metawebthree.common.enums.ResponseStatus;
import com.metawebthree.wallet.application.dto.WalletDTO;
import com.metawebthree.wallet.application.query.WalletQueryService;
import org.springframework.web.bind.annotation.*;

import java.util.Map;

@RestController
@RequestMapping("/api/admin/wallet")
public class WalletAdminController {

    private WalletQueryService queryService;

    @GetMapping("/list")
    public ApiResponse<Map<String, Object>> listWallets(
            @RequestParam(defaultValue = "1") Integer pageNum,
            @RequestParam(defaultValue = "10") Integer pageSize,
            @RequestParam(required = false) String userId,
            @RequestParam(required = false) String chainType,
            @RequestParam(required = false) String status) {

        Map<String, Object> result = queryService.listWallets(pageNum, pageSize, userId, chainType, status);
        return ApiResponse.success(result);
    }

    @GetMapping("/{id}")
    public ApiResponse<WalletDTO> getWalletById(@PathVariable Long id) {
        WalletDTO wallet = queryService.getById(id);
        return ApiResponse.success(wallet);
    }

    @GetMapping("/statistics")
    public ApiResponse<Map<String, Object>> getStatistics() {
        Map<String, Object> statistics = queryService.getStatistics();
        return ApiResponse.success(statistics);
    }

    @PostMapping("/{id}/freeze")
    public ApiResponse<WalletDTO> freezeWallet(@PathVariable Long id) {
        // Placeholder for freeze operation
        return ApiResponse.error(ResponseStatus.NOT_FOUND, "Freeze operation not implemented");
    }

    @PostMapping("/{id}/unfreeze")
    public ApiResponse<WalletDTO> unfreezeWallet(@PathVariable Long id) {
        // Placeholder for unfreeze operation
        return ApiResponse.error(ResponseStatus.NOT_FOUND, "Unfreeze operation not implemented");
    }
}