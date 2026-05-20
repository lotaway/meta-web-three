package com.metawebthree.finance.domain.repository;

import com.metawebthree.finance.domain.entity.Voucher;
import java.util.List;
import java.util.Optional;

public interface VoucherRepository {
    Optional<Voucher> findById(Long id);
    Optional<Voucher> findByVoucherNo(String voucherNo);
    List<Voucher> findByStatus(Voucher.VoucherStatus status);
    List<Voucher> findByVoucherDateBetween(java.time.LocalDateTime start, java.time.LocalDateTime end);
    List<Voucher> findByCreatedBy(String createdBy);
    List<Voucher> findAll();
    void save(Voucher voucher);
    void update(Voucher voucher);
    void delete(Long id);
}