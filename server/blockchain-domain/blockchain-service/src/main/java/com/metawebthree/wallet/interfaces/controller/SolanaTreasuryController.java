package com.metawebthree.wallet.interfaces.controller;

import com.metawebthree.common.dto.ApiResponse;
import com.metawebthree.wallet.application.service.SolanaTreasuryService;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.util.Map;

@SolanaEnabled
@RestController
@RequestMapping("/api/v1/solana/treasury")
public class SolanaTreasuryController {

    private final SolanaTreasuryService treasuryService;

    public SolanaTreasuryController(SolanaTreasuryService treasuryService) {
        this.treasuryService = treasuryService;
    }

    @PostMapping("/initialize")
    public ApiResponse<String> initialize(@RequestBody Map<String, String> request) {
        return ApiResponse.success(treasuryService.initialize(
            request.get("signerAddress"), request.get("mintAddress")));
    }

    @PostMapping("/deposit")
    public ApiResponse<String> deposit(@RequestBody Map<String, Object> request) {
        Long amount = request.get("amount") != null ? ((Number) request.get("amount")).longValue() : null;
        return ApiResponse.success(treasuryService.deposit(
            (String) request.get("signerAddress"), (String) request.get("mintAddress"), amount));
    }

    @PostMapping("/withdraw")
    public ApiResponse<String> withdraw(@RequestBody Map<String, Object> request) {
        Long amount = request.get("amount") != null ? ((Number) request.get("amount")).longValue() : null;
        return ApiResponse.success(treasuryService.withdraw(
            (String) request.get("signerAddress"), (String) request.get("mintAddress"), amount));
    }
}
