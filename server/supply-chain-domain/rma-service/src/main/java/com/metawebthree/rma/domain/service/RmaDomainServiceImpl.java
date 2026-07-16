package com.metawebthree.rma.domain.service;

import com.metawebthree.rma.domain.entity.RmaDisposition;
import com.metawebthree.rma.domain.entity.RmaInspection;
import com.metawebthree.rma.domain.entity.RmaOrder;
import com.metawebthree.rma.domain.entity.RmaOrderItem;
import com.metawebthree.rma.domain.repository.RmaDispositionRepository;
import com.metawebthree.rma.domain.repository.RmaInspectionRepository;
import com.metawebthree.rma.domain.repository.RmaOrderItemRepository;
import com.metawebthree.rma.domain.repository.RmaOrderRepository;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.math.BigDecimal;
import java.time.LocalDate;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.List;
import java.util.Optional;
import java.util.concurrent.atomic.AtomicLong;

@Service
public class RmaDomainServiceImpl implements RmaDomainService {

    private final RmaOrderRepository rmaOrderRepository;
    private final RmaOrderItemRepository rmaOrderItemRepository;
    private final RmaInspectionRepository rmaInspectionRepository;
    private final RmaDispositionRepository rmaDispositionRepository;

    private static final AtomicLong SEQ = new AtomicLong(1);
    private static final Object SEQ_LOCK = new Object();
    private static String SEQ_DATE = LocalDate.now().format(DateTimeFormatter.ofPattern("yyyyMMdd"));

    public RmaDomainServiceImpl(RmaOrderRepository rmaOrderRepository,
                                RmaOrderItemRepository rmaOrderItemRepository,
                                RmaInspectionRepository rmaInspectionRepository,
                                RmaDispositionRepository rmaDispositionRepository) {
        this.rmaOrderRepository = rmaOrderRepository;
        this.rmaOrderItemRepository = rmaOrderItemRepository;
        this.rmaInspectionRepository = rmaInspectionRepository;
        this.rmaDispositionRepository = rmaDispositionRepository;
    }

    @Override
    @Transactional
    public RmaOrder createRmaOrder(String orderNo, Long customerId, String customerName,
                                   String contactPhone, String reasonCode, String reasonDescription,
                                   Long warehouseId, String returnType, String createdBy,
                                   List<RmaOrderItem> items) {
        RmaOrder order = new RmaOrder();
        order.setRmaNo(generateRmaNo());
        order.setOrderNo(orderNo);
        order.setReturnType(returnType);
        order.setStatus("PENDING");
        order.setCustomerId(customerId);
        order.setCustomerName(customerName);
        order.setContactPhone(contactPhone);
        order.setReasonCode(reasonCode);
        order.setReasonDescription(reasonDescription);
        order.setWarehouseId(warehouseId);
        order.setTotalQuantity(items.stream().mapToInt(RmaOrderItem::getExpectedQuantity).sum());
        order.setTotalAmount(items.stream()
                .map(i -> i.getUnitPrice().multiply(BigDecimal.valueOf(i.getExpectedQuantity())))
                .reduce(BigDecimal.ZERO, BigDecimal::add));
        order.setCurrency("CNY");
        order.setCreatedBy(createdBy);
        order.setCreatedAt(LocalDateTime.now());
        order.setUpdatedAt(LocalDateTime.now());
        order.setVersion(0);

        RmaOrder saved = rmaOrderRepository.save(order);

        for (RmaOrderItem item : items) {
            item.setRmaId(saved.getId());
            item.setCreatedAt(LocalDateTime.now());
            rmaOrderItemRepository.save(item);
        }

        return saved;
    }

    @Override
    @Transactional
    public RmaOrder submitForInspection(Long rmaId) {
        RmaOrder order = rmaOrderRepository.findById(rmaId)
                .orElseThrow(() -> new IllegalArgumentException("RMA order not found: " + rmaId));
        if (!order.canInspect()) {
            throw new IllegalStateException("RMA order cannot be submitted for inspection, current status: " + order.getStatus());
        }
        order.setStatus("AWAITING_INSPECTION");
        order.setUpdatedAt(LocalDateTime.now());
        return rmaOrderRepository.save(order);
    }

    @Override
    @Transactional
    public RmaInspection recordInspection(Long rmaId, RmaInspection inspection) {
        RmaOrder order = rmaOrderRepository.findById(rmaId)
                .orElseThrow(() -> new IllegalArgumentException("RMA order not found: " + rmaId));
        if (!"AWAITING_INSPECTION".equals(order.getStatus())) {
            throw new IllegalStateException("RMA order is not awaiting inspection, current status: " + order.getStatus());
        }

        inspection.setRmaId(rmaId);
        inspection.setRmaNo(order.getRmaNo());
        inspection.setInspectionDate(LocalDateTime.now());
        inspection.setCreatedAt(LocalDateTime.now());
        inspection.setUpdatedAt(LocalDateTime.now());
        RmaInspection saved = rmaInspectionRepository.save(inspection);

        List<RmaOrderItem> items = rmaOrderItemRepository.findByRmaId(rmaId);
        for (RmaOrderItem item : items) {
            item.setInspectedQuantity(inspection.getTotalInspected() / items.size());
            item.setAcceptedQuantity(inspection.getTotalPassed() / items.size());
            rmaOrderItemRepository.save(item);
        }

        order.setStatus("INSPECTED");
        order.setUpdatedAt(LocalDateTime.now());
        rmaOrderRepository.save(order);

        return saved;
    }

    @Override
    @Transactional
    public RmaDisposition makeDisposition(Long rmaId, RmaDisposition disposition) {
        RmaOrder order = rmaOrderRepository.findById(rmaId)
                .orElseThrow(() -> new IllegalArgumentException("RMA order not found: " + rmaId));
        if (!order.canDispose()) {
            throw new IllegalStateException("RMA order cannot be disposed, current status: " + order.getStatus());
        }

        disposition.setRmaId(rmaId);
        disposition.setRmaNo(order.getRmaNo());
        disposition.setDispositionDate(LocalDateTime.now());
        disposition.setCreatedAt(LocalDateTime.now());
        disposition.setUpdatedAt(LocalDateTime.now());
        RmaDisposition saved = rmaDispositionRepository.save(disposition);

        order.setStatus("AWAITING_DISPOSITION");
        order.setUpdatedAt(LocalDateTime.now());
        rmaOrderRepository.save(order);

        return saved;
    }

    @Override
    @Transactional
    public RmaOrder executeDisposition(Long rmaId) {
        RmaOrder order = rmaOrderRepository.findById(rmaId)
                .orElseThrow(() -> new IllegalArgumentException("RMA order not found: " + rmaId));
        if (!"AWAITING_DISPOSITION".equals(order.getStatus())) {
            throw new IllegalStateException("RMA order has no pending disposition, current status: " + order.getStatus());
        }

        order.setStatus("DISPOSED");
        order.setUpdatedAt(LocalDateTime.now());
        return rmaOrderRepository.save(order);
    }

    @Override
    @Transactional
    public RmaOrder completeRmaOrder(Long rmaId) {
        RmaOrder order = rmaOrderRepository.findById(rmaId)
                .orElseThrow(() -> new IllegalArgumentException("RMA order not found: " + rmaId));
        if (!order.canComplete()) {
            throw new IllegalStateException("RMA order cannot be completed, current status: " + order.getStatus());
        }
        order.setStatus("COMPLETED");
        order.setUpdatedAt(LocalDateTime.now());
        return rmaOrderRepository.save(order);
    }

    @Override
    @Transactional
    public RmaOrder cancelRmaOrder(Long rmaId) {
        RmaOrder order = rmaOrderRepository.findById(rmaId)
                .orElseThrow(() -> new IllegalArgumentException("RMA order not found: " + rmaId));
        if (!order.canCancel()) {
            throw new IllegalStateException("RMA order cannot be cancelled, current status: " + order.getStatus());
        }
        order.setStatus("CANCELLED");
        order.setUpdatedAt(LocalDateTime.now());
        return rmaOrderRepository.save(order);
    }

    @Override
    public Optional<RmaOrder> getRmaOrder(Long rmaId) {
        return rmaOrderRepository.findById(rmaId);
    }

    @Override
    public Optional<RmaOrder> getRmaOrderByNo(String rmaNo) {
        return rmaOrderRepository.findByRmaNo(rmaNo);
    }

    private String generateRmaNo() {
        String datePart = LocalDate.now().format(DateTimeFormatter.ofPattern("yyyyMMdd"));
        synchronized (SEQ_LOCK) {
            if (!datePart.equals(SEQ_DATE)) {
                SEQ.set(1);
                SEQ_DATE = datePart;
            }
            long seq = SEQ.getAndIncrement();
            return "RMA" + datePart + String.format("%06d", seq % 1000000);
        }
    }
}
