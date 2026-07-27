package com.metawebthree.wallet.interfaces.controller;

import com.metawebthree.common.dto.ApiResponse;
import com.metawebthree.wallet.application.dto.CreateTokenRequest;
import com.metawebthree.wallet.application.dto.MintTokenRequest;
import com.metawebthree.wallet.application.dto.SolanaTokenDTO;
import com.metawebthree.wallet.application.service.SolanaTokenService;
import org.springframework.web.bind.annotation.*;

import java.util.Map;

@RestController
@RequestMapping("/api/v1/solana/tokens")
public class SolanaTokenController {

    private final SolanaTokenService solanaTokenService;

    public SolanaTokenController(SolanaTokenService solanaTokenService) {
        this.solanaTokenService = solanaTokenService;
    }

    @PostMapping
    public ApiResponse<SolanaTokenDTO> createToken(@RequestBody CreateTokenRequest request) {
        SolanaTokenDTO result = solanaTokenService.createToken(request);
        return ApiResponse.success(result);
    }

    @GetMapping("/{mintAddress}")
    public ApiResponse<SolanaTokenDTO> getToken(@PathVariable String mintAddress) {
        SolanaTokenDTO result = solanaTokenService.getToken(mintAddress);
        return ApiResponse.success(result);
    }

    @PostMapping("/{mintAddress}/mint")
    public ApiResponse<SolanaTokenDTO> mintTo(
            @PathVariable String mintAddress,
            @RequestBody Map<String, Object> request) {
        MintTokenRequest mintRequest = new MintTokenRequest();
        mintRequest.setMintAddress(mintAddress);
        mintRequest.setRecipient((String) request.get("recipient"));
        mintRequest.setAmount(request.get("amount") != null ? ((Number) request.get("amount")).longValue() : null);
        SolanaTokenDTO result = solanaTokenService.mintTo(mintRequest);
        return ApiResponse.success(result);
    }

    @PostMapping("/{mintAddress}/burn")
    public ApiResponse<String> burnTokens(
            @PathVariable String mintAddress,
            @RequestBody Map<String, Object> request) {
        Long amount = request.get("amount") != null ? ((Number) request.get("amount")).longValue() : null;
        String ownerAddress = (String) request.get("ownerAddress");
        String result = solanaTokenService.burnTokens(mintAddress, amount, ownerAddress);
        return ApiResponse.success(result);
    }
}
