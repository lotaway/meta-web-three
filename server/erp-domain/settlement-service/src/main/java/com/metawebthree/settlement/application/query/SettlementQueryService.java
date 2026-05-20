package com.metawebthree.settlement.application.query;

import com.metawebthree.settlement.domain.entity.SettlementOrder;
import com.metawebthree.settlement.domain.repository.SettlementOrderRepository;
import org.springframework.stereotype.Service;
import java.time.LocalDateTime;
import java.util.List;
import java.util.Optional;

@Service
public class SettlementQueryService {
    private final SettlementOrderRepository repository;

    public SettlementQueryService(SettlementOrderRepository repository) {
        this.repository = repository;
    }

    public Optional<SettlementOrder> getById(Long id) {
        return repository.findById(id);
    }

    public Optional<SettlementOrder> getBySettlementNo(String settlementNo) {
        return repository.findBySettlementNo(settlementNo);
    }

    public List<SettlementOrder> listByStatus(String status) {
        SettlementOrder.SettlementStatus s = SettlementOrder.SettlementStatus.valueOf(status.toUpperCase());
        return repository.findByStatus(s);
    }

    public List<SettlementOrder> listByMerchantId(Long merchantId) {
        return repository.findByMerchantId(merchantId);
    }

    public List<SettlementOrder> listByDateRange(LocalDateTime start, LocalDateTime end) {
        return repository.findByDateRange(start, end);
    }

    public List<SettlementOrder> listAll() {
        return repository.findAll();
    }
}