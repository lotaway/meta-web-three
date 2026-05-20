package com.metawebthree.finance.application.query;

import com.metawebthree.finance.domain.entity.Account;
import com.metawebthree.finance.domain.repository.AccountRepository;
import org.springframework.stereotype.Service;
import java.util.List;
import java.util.Optional;

@Service
public class AccountQueryService {
    private final AccountRepository accountRepository;

    public AccountQueryService(AccountRepository accountRepository) {
        this.accountRepository = accountRepository;
    }

    public Optional<Account> getById(Long accountId) {
        return accountRepository.findById(accountId);
    }

    public Optional<Account> getByAccountNo(String accountNo) {
        return accountRepository.findByAccountNo(accountNo);
    }

    public List<Account> listActiveAccounts() {
        return accountRepository.findByStatus(Account.AccountStatus.ACTIVE);
    }

    public List<Account> listAllAccounts() {
        return accountRepository.findAll();
    }

    public List<Account> listByType(String type) {
        Account.AccountType accountType = Account.AccountType.valueOf(type.toUpperCase());
        return accountRepository.findByType(accountType);
    }
}