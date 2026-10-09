package com.metawebthree.finance.infrastructure.persistence.repository.cash;

import com.baomidou.mybatisplus.core.toolkit.IdWorker;
import com.metawebthree.finance.domain.entity.cash.CashTransfer;
import com.metawebthree.finance.domain.repository.cash.CashTransferRepository;
import org.springframework.stereotype.Repository;

import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.Collectors;

/**
 * 资金划转仓储（本地开发用内存实现）。
 */
@Repository
public class CashTransferRepositoryImpl implements CashTransferRepository {

    private final Map<Long, CashTransfer> store = new ConcurrentHashMap<>();

    @Override
    public Long save(CashTransfer transfer) {
        if (transfer.getId() == null) {
            transfer.setId(IdWorker.getId());
        }
        store.put(transfer.getId(), transfer);
        return transfer.getId();
    }

    @Override
    public Optional<CashTransfer> findById(Long id) {
        return Optional.ofNullable(store.get(id));
    }

    @Override
    public Optional<CashTransfer> findByTransferNo(String transferNo) {
        return store.values().stream()
                .filter(t -> Objects.equals(t.getTransferNo(), transferNo))
                .findFirst();
    }

    @Override
    public List<CashTransfer> findAll() {
        return store.values().stream().collect(Collectors.toList());
    }

    @Override
    public List<CashTransfer> findByStatus(CashTransfer.CashTransferStatus status) {
        return store.values().stream()
                .filter(t -> Objects.equals(t.getStatus(), status))
                .collect(Collectors.toList());
    }

    @Override
    public List<CashTransfer> findByFromAccountId(Long fromAccountId) {
        return store.values().stream()
                .filter(t -> Objects.equals(t.getFromAccountId(), fromAccountId))
                .collect(Collectors.toList());
    }

    @Override
    public List<CashTransfer> findByToAccountId(Long toAccountId) {
        return store.values().stream()
                .filter(t -> Objects.equals(t.getToAccountId(), toAccountId))
                .collect(Collectors.toList());
    }

    @Override
    public List<CashTransfer> findByFromAccountIdOrToAccountId(Long fromAccountId, Long toAccountId) {
        return store.values().stream()
                .filter(t -> Objects.equals(t.getFromAccountId(), fromAccountId)
                        || Objects.equals(t.getToAccountId(), toAccountId))
                .collect(Collectors.toList());
    }

    @Override
    public void update(CashTransfer transfer) {
        if (transfer.getId() != null) {
            store.put(transfer.getId(), transfer);
        }
    }

    @Override
    public void deleteById(Long id) {
        store.remove(id);
    }
}
