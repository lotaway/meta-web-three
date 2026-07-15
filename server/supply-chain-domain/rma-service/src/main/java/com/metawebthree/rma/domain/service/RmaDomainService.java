package com.metawebthree.rma.domain.service;

import com.metawebthree.rma.domain.entity.RmaDisposition;
import com.metawebthree.rma.domain.entity.RmaInspection;
import com.metawebthree.rma.domain.entity.RmaOrder;
import com.metawebthree.rma.domain.entity.RmaOrderItem;
import java.util.List;
import java.util.Optional;

public interface RmaDomainService {

    RmaOrder createRmaOrder(String orderNo, Long customerId, String customerName,
                            String contactPhone, String reasonCode, String reasonDescription,
                            Long warehouseId, String returnType, String createdBy,
                            List<RmaOrderItem> items);

    RmaOrder submitForInspection(Long rmaId);

    RmaInspection recordInspection(Long rmaId, RmaInspection inspection);

    RmaDisposition makeDisposition(Long rmaId, RmaDisposition disposition);

    RmaOrder executeDisposition(Long rmaId);

    RmaOrder cancelRmaOrder(Long rmaId);

    RmaOrder completeRmaOrder(Long rmaId);

    Optional<RmaOrder> getRmaOrder(Long rmaId);

    Optional<RmaOrder> getRmaOrderByNo(String rmaNo);
}
