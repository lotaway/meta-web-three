package com.metawebthree.wallet.interfaces.controller;

import com.metawebthree.common.dto.ApiResponse;
import com.metawebthree.wallet.application.dto.CommissionDTO;
import com.metawebthree.wallet.application.service.SolanaCommissionService;
import org.springframework.web.bind.annotation.*;

import java.util.Map;

@RestController
@RequestMapping("/api/v1/solana/commission")
public class SolanaCommissionController {

    private final SolanaCommissionService commissionService;

    public SolanaCommissionController(SolanaCommissionService commissionService) {
        this.commissionService = commissionService;
    }

    @PostMapping("/upline")
    public ApiResponse<CommissionDTO> setUpline(@RequestBody Map<String, String> request) {
        String target = request.get("target");
        String upline = request.get("upline");
        CommissionDTO result = commissionService.setUpline(target, upline);
        return ApiResponse.success(result);
    }

    @GetMapping("/{account}")
    public ApiResponse<CommissionDTO> getCommissionGraph(@PathVariable String account) {
        CommissionDTO result = commissionService.getCommissionGraph(account);
        return ApiResponse.success(result);
    }

    @PostMapping("/distribute")
    public ApiResponse<String> distributeCommission(@RequestBody Map<String, Object> request) {
        String seller = (String) request.get("seller");
        String paymentMint = (String) request.get("paymentMint");
        Long saleAmount = request.get("saleAmount") != null ? ((Number) request.get("saleAmount")).longValue() : null;
        String result = commissionService.distributeCommission(seller, paymentMint, saleAmount);
        return ApiResponse.success(result);
    }
}
