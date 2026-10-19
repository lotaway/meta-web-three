package com.metawebthree.rma.domain.repository;

import com.metawebthree.rma.domain.entity.RmaStockInRecord;

import java.util.List;
import java.util.Optional;

public interface RmaStockInRecordRepository {

    Optional<RmaStockInRecord> findByRmaIdAndSkuCode(Long rmaId, String skuCode);

    RmaStockInRecord save(RmaStockInRecord record);

    /** Records that still need to be delivered to inventory, oldest first. */
    List<RmaStockInRecord> findByStatusIn(List<String> statuses);
}
