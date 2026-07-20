package com.metawebthree.rma.domain.service;

import com.metawebthree.rma.domain.entity.RmaDisposition;
import com.metawebthree.rma.domain.entity.RmaInspection;
import com.metawebthree.rma.domain.entity.RmaOrder;
import com.metawebthree.rma.domain.entity.RmaOrderItem;
import com.metawebthree.rma.domain.entity.ReturnShipping;
import java.util.List;
import java.util.Optional;

public interface RmaDomainService {

    RmaOrder createRmaOrder(String orderNo, Long customerId, String customerName,
                            String contactPhone, String reasonCode, String reasonDescription,
                            Long warehouseId, String returnType, String createdBy,
                            List<RmaOrderItem> items);

    void saveRmaOrder(RmaOrder order);

    void saveRmaOrder(RmaOrder order, List<RmaOrderItem> items);

    void saveInspection(RmaInspection inspection);

    void saveDisposition(RmaDisposition disposition);

    ReturnShipping saveReturnShipping(ReturnShipping shipping);

    RmaOrder submitForInspection(Long rmaId);

    RmaInspection recordInspection(RmaOrder order, RmaInspection inspection);

    RmaDisposition makeDisposition(RmaOrder order, RmaDisposition disposition);

    RmaOrder executeDisposition(Long rmaId);

    RmaOrder completeRmaOrder(Long rmaId);

    RmaOrder cancelRmaOrder(Long rmaId);

    Optional<RmaOrder> getRmaOrder(Long rmaId);

    Optional<RmaOrder> getRmaOrderByNo(String rmaNo);
}
