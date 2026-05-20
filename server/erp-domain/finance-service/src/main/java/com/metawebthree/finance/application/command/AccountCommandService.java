package com.metawebthree.finance.application.command;

import com.metawebthree.finance.domain.entity.Account;
import com.metawebthree.finance.domain.repository.AccountRepository;
import com.metawebthree.finance.infrastructure.event.FinanceEventPublisher;
import org.springframework.stereotype.Service;
import java.math.BigDecimal;
import java.util.List;

@Service
public class AccountCommandService {
    private final AccountRepository accountRepository;
    private final FinanceEventPublisher eventPublisher;

    public AccountCommandService(AccountRepository accountRepository, FinanceEventPublisher eventPublisher) {
        this.accountRepository = accountRepository;
        this.eventPublisher = eventPublisher;
    }

    public Long createAccount(String accountNo, String accountName, String type) {
        Account.AccountType accountType = Account.AccountType.valueOf(type.toUpperCase());
        Account account = new Account();
        account.create(accountNo, accountName, accountType);
        accountRepository.save(account);
        eventPublisher.publishAccountCreated(account.getId(), accountNo, accountName, type);
        return account.getId();
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

    public void credit(Long accountId, BigDecimal amount) {
        Account account = accountRepository.findById(accountId)
            .orElseThrow(() -> new IllegalArgumentException("Account not found"));
        BigDecimal beforeAmount = account.getBalance();
        account.credit(amount);
        accountRepository.update(account);
        eventPublisher.publishAccountBalanceChanged(
            accountId, account.getAccountNo(), beforeAmount, account.getBalance(), "CREDIT");
    }

    public void debit(Long accountId, BigDecimal amount) {
        Account account = accountRepository.findById(accountId)
            .orElseThrow(() -> new IllegalArgumentException("Account not found"));
        BigDecimal beforeAmount = account.getBalance();
        account.debit(amount);
        accountRepository.update(account);
        eventPublisher.publishAccountBalanceChanged(
            accountId, account.getAccountNo(), beforeAmount, account.getBalance(), "DEBIT");
    }
}