package com.metawebthree.wallet.interfaces.controller;

import com.metawebthree.common.annotations.RequirePermission;
import com.metawebthree.wallet.application.command.WalletCommandService;
import com.metawebthree.wallet.application.dto.WalletDTO;
import com.metawebthree.wallet.application.query.WalletQueryService;
import com.metawebthree.wallet.domain.model.WalletPermission;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.math.BigDecimal;
import java.util.Map;

@RestController
@RequestMapping("/api/v1/wallets")
public class WalletController {

    private final WalletCommandService commandService;
    private final WalletQueryService queryService;

    public WalletController(WalletCommandService commandService, WalletQueryService queryService) {
        this.commandService = commandService;
        this.queryService = queryService;
    }

    @PostMapping
    @RequirePermission(WalletPermission.WALLET_CREATE)
    public ResponseEntity<WalletDTO> createWallet(@RequestBody Map<String, String> request) {
        String userId = request.get("userId");
        String chainType = request.get("chainType");
        String address = request.get("address");
        WalletDTO wallet = commandService.createWallet(userId, chainType, address);
        return ResponseEntity.status(HttpStatus.CREATED).body(wallet);
    }

    @GetMapping("/{id}")
    @RequirePermission(WalletPermission.WALLET_READ)
    public ResponseEntity<WalletDTO> getWallet(@PathVariable Long id, @RequestParam(required = false) String userId) {
        WalletDTO wallet = queryService.getById(id, userId);
        if (wallet == null) {
            return ResponseEntity.status(HttpStatus.NOT_FOUND).build();
        }
        return ResponseEntity.ok(wallet);
    }

    @PostMapping("/{id}/deposit")
    @RequirePermission(WalletPermission.WALLET_DEPOSIT)
    public ResponseEntity<WalletDTO> deposit(@PathVariable Long id, @RequestBody Map<String, BigDecimal> request,
                                           @RequestParam(required = false) String userId) {
        BigDecimal amount = request.get("amount");
        WalletDTO wallet = commandService.deposit(id, amount, userId);
        return ResponseEntity.ok(wallet);
    }

    @PostMapping("/{id}/withdraw")
    @RequirePermission(WalletPermission.WALLET_WITHDRAW)
    public ResponseEntity<WalletDTO> withdraw(@PathVariable Long id, @RequestBody Map<String, BigDecimal> request,
                                            @RequestParam(required = false) String userId) {
        BigDecimal amount = request.get("amount");
        WalletDTO wallet = commandService.withdraw(id, amount, userId);
        return ResponseEntity.ok(wallet);
    }
}