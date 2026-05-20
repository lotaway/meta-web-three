package com.metawebthree.finance.domain.service;

import com.metawebthree.finance.domain.entity.Account;
import com.metawebthree.finance.domain.repository.AccountRepository;
import java.util.List;

public class AccountDomainService {
    private final AccountRepository accountRepository;

    public AccountDomainService(AccountRepository accountRepository) {
        this.accountRepository = accountRepository;
    }

    public Account createAccount(String accountNo, String accountName, Account.AccountType type) {
        accountRepository.findByAccountNo(accountNo).ifPresent(existing -> {
            throw new IllegalArgumentException("Account number already exists");
        });
        Account account = new Account();
        account.create(accountNo, accountName, type);
        accountRepository.save(account);
        return account;
    }

    public void freezeAccount(Long accountId) {
        Account account = accountRepository.findById(accountId)
            .orElseThrow(() -> new IllegalArgumentException("Account not found"));
        account.freeze();
        accountRepository.update(account);
    }

    public void unfreezeAccount(Long accountId) {
        Account account = accountRepository.findById(accountId)
            .orElseThrow(() -> new IllegalArgumentException("Account not found"));
        account.unfreeze();
        accountRepository.update(account);
    }

    public void closeAccount(Long accountId) {
        Account account = accountRepository.findById(accountId)
            .orElseThrow(() -> new IllegalArgumentException("Account not found"));
        account.close();
        accountRepository.update(account);
    }

    public void credit(Long accountId, java.math.BigDecimal amount) {
        Account account = accountRepository.findById(accountId)
            .orElseThrow(() -> new IllegalArgumentException("Account not found"));
        account.credit(amount);
        accountRepository.update(account);
    }

    public void debit(Long accountId, java.math.BigDecimal amount) {
        Account account = accountRepository.findById(accountId)
            .orElseThrow(() -> new IllegalArgumentException("Account not found"));
        account.debit(amount);
        accountRepository.update(account);
    }

    public List<Account> listActiveAccounts() {
        return accountRepository.findByStatus(Account.AccountStatus.ACTIVE);
    }

    public Account getAccount(Long accountId) {
        return accountRepository.findById(accountId)
            .orElseThrow(() -> new IllegalArgumentException("Account not found"));
    }
}