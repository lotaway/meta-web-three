package com.metawebthree.finance.infrastructure.persistence.repository;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.metawebthree.finance.domain.entity.Account;
import com.metawebthree.finance.domain.repository.AccountRepository;
import com.metawebthree.finance.infrastructure.persistence.converter.AccountConverter;
import com.metawebthree.finance.infrastructure.persistence.dataobject.AccountDO;
import com.metawebthree.finance.infrastructure.persistence.mapper.AccountMapper;
import org.springframework.stereotype.Repository;

import java.util.List;
import java.util.Optional;
import java.util.stream.Collectors;

@Repository
public class AccountRepositoryImpl implements AccountRepository {

    private final AccountMapper accountMapper;
    private final AccountConverter accountConverter;

    public AccountRepositoryImpl(AccountMapper accountMapper, AccountConverter accountConverter) {
        this.accountMapper = accountMapper;
        this.accountConverter = accountConverter;
    }

    @Override
    public Optional<Account> findById(Long id) {
        AccountDO accountDO = accountMapper.selectById(id);
        return Optional.ofNullable(accountConverter.toEntity(accountDO));
    }

    @Override
    public Optional<Account> findByAccountNo(String accountNo) {
        LambdaQueryWrapper<AccountDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(AccountDO::getAccountNo, accountNo);
        AccountDO accountDO = accountMapper.selectOne(wrapper);
        return Optional.ofNullable(accountConverter.toEntity(accountDO));
    }

    @Override
    public List<Account> findByStatus(Account.AccountStatus status) {
        LambdaQueryWrapper<AccountDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(AccountDO::getStatus, status.name());
        List<AccountDO> accountDOs = accountMapper.selectList(wrapper);
        return accountDOs.stream()
                .map(accountConverter::toEntity)
                .collect(Collectors.toList());
    }

    @Override
    public List<Account> findByType(Account.AccountType type) {
        LambdaQueryWrapper<AccountDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(AccountDO::getType, type.name());
        List<AccountDO> accountDOs = accountMapper.selectList(wrapper);
        return accountDOs.stream()
                .map(accountConverter::toEntity)
                .collect(Collectors.toList());
    }

    @Override
    public List<Account> findAll() {
        List<AccountDO> accountDOs = accountMapper.selectList(null);
        return accountDOs.stream()
                .map(accountConverter::toEntity)
                .collect(Collectors.toList());
    }

    @Override
    public void save(Account account) {
        AccountDO accountDO = accountConverter.toDO(account);
        if (account.getId() == null) {
            accountMapper.insert(accountDO);
            account.setId(accountDO.getId());
        } else {
            accountMapper.updateById(accountDO);
        }
    }

    @Override
    public void update(Account account) {
        AccountDO accountDO = accountConverter.toDO(account);
        accountMapper.updateById(accountDO);
    }

    @Override
    public void delete(Long id) {
        accountMapper.deleteById(id);
    }
}