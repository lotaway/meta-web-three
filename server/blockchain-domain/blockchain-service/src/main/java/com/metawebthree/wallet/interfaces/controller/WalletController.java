package com.metawebthree.wallet.interfaces.controller;

import com.metawebthree.wallet.application.command.WalletCommandService;
import com.metawebthree.wallet.application.dto.WalletDTO;
import com.metawebthree.wallet.application.query.WalletQueryService;
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
    public ResponseEntity<WalletDTO> createWallet(@RequestBody Map<String, String> request) {
        String userId = request.get("userId");
        String chainType = request.get("chainType");
        String address = request.get("address");
        WalletDTO wallet = commandService.createWallet(userId, chainType, address);
        return ResponseEntity.ok(wallet);
    }

    @GetMapping("/{id}")
    public ResponseEntity<WalletDTO> getWallet(@PathVariable Long id) {
        return ResponseEntity.ok(queryService.getById(id));
    }

    @PostMapping("/{id}/deposit")
    public ResponseEntity<WalletDTO> deposit(@PathVariable Long id, @RequestBody Map<String, BigDecimal> request) {
        BigDecimal amount = request.get("amount");
        return ResponseEntity.ok(commandService.deposit(id, amount));
    }

    @PostMapping("/{id}/withdraw")
    public ResponseEntity<WalletDTO> withdraw(@PathVariable Long id, @RequestBody Map<String, BigDecimal> request) {
        BigDecimal amount = request.get("amount");
        return ResponseEntity.ok(commandService.withdraw(id, amount));
    }
}