package com.metawebthree.finance.interfaces.controller;

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

    @PostMapping
    public ResponseEntity<Long> createAccount(@RequestBody AccountCreateRequest request) {
        Long id = commandService.createAccount(request.getAccountNo(), request.getAccountName(), request.getType());
        return ResponseEntity.ok(id);
    }

    @GetMapping("/{id}")
    public ResponseEntity<Account> getAccount(@PathVariable Long id) {
        return queryService.getById(id)
            .map(ResponseEntity::ok)
            .orElse(ResponseEntity.notFound().build());
    }

    @GetMapping
    public ResponseEntity<List<Account>> listAccounts(@RequestParam(required = false) String status,
                                                       @RequestParam(required = false) String type) {
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

    @PostMapping("/{id}/freeze")
    public ResponseEntity<Void> freezeAccount(@PathVariable Long id) {
        commandService.freezeAccount(id);
        return ResponseEntity.ok().build();
    }

    @PostMapping("/{id}/unfreeze")
    public ResponseEntity<Void> unfreezeAccount(@PathVariable Long id) {
        commandService.unfreezeAccount(id);
        return ResponseEntity.ok().build();
    }

    @PostMapping("/{id}/close")
    public ResponseEntity<Void> closeAccount(@PathVariable Long id) {
        commandService.closeAccount(id);
        return ResponseEntity.ok().build();
    }

    @PostMapping("/{id}/credit")
    public ResponseEntity<Void> credit(@PathVariable Long id, @RequestParam BigDecimal amount) {
        commandService.credit(id, amount);
        return ResponseEntity.ok().build();
    }

    @PostMapping("/{id}/debit")
    public ResponseEntity<Void> debit(@PathVariable Long id, @RequestParam BigDecimal amount) {
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