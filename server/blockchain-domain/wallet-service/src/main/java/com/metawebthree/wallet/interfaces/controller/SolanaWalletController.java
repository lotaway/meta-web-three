package com.metawebthree.wallet.interfaces.controller;

import com.metawebthree.common.dto.ApiResponse;
import com.metawebthree.wallet.infrastructure.solana.SolanaWalletManager;
import org.springframework.web.bind.annotation.*;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

@RestController
@RequestMapping("/api/v1/solana/wallets")
public class SolanaWalletController {

    private final SolanaWalletManager walletManager;

    public SolanaWalletController(SolanaWalletManager walletManager) {
        this.walletManager = walletManager;
    }

    @PostMapping
    public ApiResponse<Map<String, String>> generateWallet() {
        String address = walletManager.generateWallet();
        Map<String, String> result = new HashMap<>();
        result.put("address", address);
        return ApiResponse.success(result);
    }

    @PostMapping("/import")
    public ApiResponse<Map<String, String>> importWallet(@RequestBody Map<String, String> request) {
        String privateKeyB58 = request.get("privateKey");
        if (privateKeyB58 == null || privateKeyB58.isBlank()) {
            return new ApiResponse<>("1002", "privateKey is required", null);
        }
        String address = walletManager.importWallet(privateKeyB58);
        Map<String, String> result = new HashMap<>();
        result.put("address", address);
        return ApiResponse.success(result);
    }

    @GetMapping
    public ApiResponse<List<String>> listWallets() {
        List<String> addresses = walletManager.listWallets();
        return ApiResponse.success(addresses);
    }
}
