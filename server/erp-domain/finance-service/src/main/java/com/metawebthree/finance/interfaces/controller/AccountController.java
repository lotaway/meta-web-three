package com.metawebthree.finance.interfaces.controller;

import com.metawebthree.common.annotations.RequirePermission;
import com.metawebthree.common.ERPPermissions;
import com.metawebthree.finance.application.command.AccountCommandService;
import com.metawebthree.finance.application.query.AccountQueryService;
import com.metawebthree.finance.domain.entity.Account;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import java.math.BigDecimal;
import java.util.List;

@RestController
@RequestMapping("/api/finance/accounts")
public class AccountController {
    private final AccountCommandService commandService;
    private final AccountQueryService queryService;

    public AccountController(AccountCommandService commandService, AccountQueryService queryService) {
        this.commandService = commandService;
        this.queryService = queryService;
    }

    @RequirePermission(ERPPermissions.ACCOUNT_CREATE)
    @PostMapping
    public ResponseEntity<Long> createAccount(@RequestBody AccountCreateRequest request,
            @RequestHeader(value = "X-User-Id", defaultValue = "system") String userId,
            @RequestHeader(value = "X-User-Role", defaultValue = "") String userRole) {
        Long id = commandService.createAccount(request.getAccountNo(), request.getAccountName(), request.getType());
        return ResponseEntity.ok(id);
    }

    @RequirePermission(ERPPermissions.ACCOUNT_READ)
    @GetMapping("/{id}")
    public ResponseEntity<Account> getAccount(@PathVariable Long id,
            @RequestHeader(value = "X-User-Id", defaultValue = "system") String userId,
            @RequestHeader(value = "X-User-Role", defaultValue = "") String userRole) {
        return queryService.getById(id)
            .map(ResponseEntity::ok)
            .orElse(ResponseEntity.notFound().build());
    }

    @RequirePermission(ERPPermissions.ACCOUNT_READ)
    @GetMapping
    public ResponseEntity<List<Account>> listAccounts(@RequestParam(required = false) String status,
                                                       @RequestParam(required = false) String type,
            @RequestHeader(value = "X-User-Id", defaultValue = "system") String userId,
            @RequestHeader(value = "X-User-Role", defaultValue = "") String userRole) {
        List<Account> accounts;
        if ("ACTIVE".equalsIgnoreCase(status)) {
            accounts = queryService.listActiveAccounts();
        } else if (type != null) {
            accounts = queryService.listByType(type);
        } else {
            accounts = queryService.listAllAccounts();
        }
        return ResponseEntity.ok(accounts);
    }

    @RequirePermission(ERPPermissions.ACCOUNT_UPDATE)
    @PostMapping("/{id}/freeze")
    public ResponseEntity<Void> freezeAccount(@PathVariable Long id,
            @RequestHeader(value = "X-User-Id", defaultValue = "system") String userId,
            @RequestHeader(value = "X-User-Role", defaultValue = "") String userRole) {
        commandService.freezeAccount(id);
        return ResponseEntity.ok().build();
    }

    @RequirePermission(ERPPermissions.ACCOUNT_UPDATE)
    @PostMapping("/{id}/unfreeze")
    public ResponseEntity<Void> unfreezeAccount(@PathVariable Long id,
            @RequestHeader(value = "X-User-Id", defaultValue = "system") String userId,
            @RequestHeader(value = "X-User-Role", defaultValue = "") String userRole) {
        commandService.unfreezeAccount(id);
        return ResponseEntity.ok().build();
    }

    @RequirePermission(ERPPermissions.ACCOUNT_UPDATE)
    @PostMapping("/{id}/close")
    public ResponseEntity<Void> closeAccount(@PathVariable Long id,
            @RequestHeader(value = "X-User-Id", defaultValue = "system") String userId,
            @RequestHeader(value = "X-User-Role", defaultValue = "") String userRole) {
        commandService.closeAccount(id);
        return ResponseEntity.ok().build();
    }

    @RequirePermission(ERPPermissions.ACCOUNT_UPDATE)
    @PostMapping("/{id}/credit")
    public ResponseEntity<Void> credit(@PathVariable Long id, @RequestParam BigDecimal amount,
            @RequestHeader(value = "X-User-Id", defaultValue = "system") String userId,
            @RequestHeader(value = "X-User-Role", defaultValue = "") String userRole) {
        commandService.credit(id, amount);
        return ResponseEntity.ok().build();
    }

    @RequirePermission(ERPPermissions.ACCOUNT_UPDATE)
    @PostMapping("/{id}/debit")
    public ResponseEntity<Void> debit(@PathVariable Long id, @RequestParam BigDecimal amount,
            @RequestHeader(value = "X-User-Id", defaultValue = "system") String userId,
            @RequestHeader(value = "X-User-Role", defaultValue = "") String userRole) {
        commandService.debit(id, amount);
        return ResponseEntity.ok().build();
    }

    public static class AccountCreateRequest {
        private String accountNo;
        private String accountName;
        private String type;

        public String getAccountNo() { return accountNo; }
        public void setAccountNo(String accountNo) { this.accountNo = accountNo; }
        public String getAccountName() { return accountName; }
        public void setAccountName(String accountName) { this.accountName = accountName; }
        public String getType() { return type; }
        public void setType(String type) { this.type = type; }
    }
}