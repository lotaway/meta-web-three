package com.metawebthree.rma.domain.service;

import com.metawebthree.rma.domain.RmaOrderStatus;
import com.metawebthree.rma.domain.RmaSequenceGenerator;
import com.metawebthree.rma.domain.entity.*;
import com.metawebthree.rma.domain.repository.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.math.BigDecimal;
import java.time.LocalDateTime;
import java.util.List;
import java.util.Optional;


public class RmaDomainServiceImpl implements RmaDomainService {

    private static final Logger log = LoggerFactory.getLogger(RmaDomainServiceImpl.class);

    private static final String CURRENCY_CNY = "CNY";
    private static final String RMA_NO_PREFIX = "RMA";
    private static final String RMA_NO_FORMAT = "%06d";
    private static final int INITIAL_VERSION = 0;

    private final RmaOrderRepository rmaOrderRepository;
    private final RmaOrderItemRepository rmaOrderItemRepository;
    private final RmaInspectionRepository rmaInspectionRepository;
    private final RmaDispositionRepository rmaDispositionRepository;
    private final ReturnShippingRepository returnShippingRepository;
    private final RmaSequenceGenerator rmaSequenceGenerator;

    public RmaDomainServiceImpl(RmaOrderRepository rmaOrderRepository,
                                RmaOrderItemRepository rmaOrderItemRepository,
                                RmaInspectionRepository rmaInspectionRepository,
                                RmaDispositionRepository rmaDispositionRepository,
                                ReturnShippingRepository returnShippingRepository,
                                RmaSequenceGenerator rmaSequenceGenerator) {
        this.rmaOrderRepository = rmaOrderRepository;
        this.rmaOrderItemRepository = rmaOrderItemRepository;
        this.rmaInspectionRepository = rmaInspectionRepository;
        this.rmaDispositionRepository = rmaDispositionRepository;
        this.returnShippingRepository = returnShippingRepository;
        this.rmaSequenceGenerator = rmaSequenceGenerator;
    }

    @Override
    public RmaOrder createRmaOrder(String orderNo, Long customerId, String customerName,
                                   String contactPhone, String reasonCode, String reasonDescription,
                                   Long warehouseId, String returnType, String createdBy,
                                   List<RmaOrderItem> items) {
        RmaOrder order = buildOrder(orderNo, customerId, customerName, contactPhone,
                reasonCode, reasonDescription, warehouseId, returnType, createdBy, items);
        log.info("Built RMA order: rmaNo={}, orderNo={}, status={}", order.getRmaNo(), orderNo, order.getStatus());
        return order;
    }

    @Override
    public void saveRmaOrder(RmaOrder order) {
        rmaOrderRepository.save(order);
        log.info("Saved RMA order: id={}, rmaNo={}, status={}", order.getId(), order.getRmaNo(), order.getStatus());
    }

    @Override
    public void saveRmaOrder(RmaOrder order, List<RmaOrderItem> items) {
        RmaOrder saved = rmaOrderRepository.save(order);
        if (items != null) {
            for (RmaOrderItem item : items) {
                item.setRmaId(saved.getId());
                item.setCreatedAt(LocalDateTime.now());
                rmaOrderItemRepository.save(item);
            }
        }
        log.info("Saved RMA order with {} items: id={}, rmaNo={}", items != null ? items.size() : 0, saved.getId(), saved.getRmaNo());
    }

    @Override
    public void saveInspection(RmaInspection inspection) {
        rmaInspectionRepository.save(inspection);
        log.info("Saved inspection: id={}, rmaId={}, result={}", inspection.getId(), inspection.getRmaId(), inspection.getResult());
    }

    @Override
    public void saveDisposition(RmaDisposition disposition) {
        rmaDispositionRepository.save(disposition);
        log.info("Saved disposition: id={}, rmaId={}, type={}", disposition.getId(), disposition.getRmaId(), disposition.getDispositionType());
    }

    @Override
    public ReturnShipping saveReturnShipping(ReturnShipping shipping) {
        ReturnShipping saved = returnShippingRepository.save(shipping);
        log.info("Saved return shipping: id={}, rmaId={}, trackingNo={}", saved.getId(), saved.getRmaId(), saved.getTrackingNo());
        return saved;
    }

    @Override
    public RmaOrder submitForInspection(Long rmaId) {
        RmaOrder order = findOrderOrThrow(rmaId);
        if (!order.canInspect()) {
            throw new IllegalStateException("RMA order cannot be submitted for inspection, current status: " + order.getStatus());
        }
        order.setStatus(RmaOrderStatus.AWAITING_INSPECTION);
        order.setUpdatedAt(LocalDateTime.now());
        log.info("Submitted RMA order for inspection: id={}, rmaNo={}, newStatus={}", rmaId, order.getRmaNo(), order.getStatus());
        return order;
    }

    @Override
    public RmaInspection recordInspection(RmaOrder order, RmaInspection inspection) {
        if (RmaOrderStatus.AWAITING_INSPECTION != order.getStatus()) {
            throw new IllegalStateException("RMA order is not awaiting inspection, current status: " + order.getStatus());
        }

        inspection.setRmaId(order.getId());
        inspection.setRmaNo(order.getRmaNo());
        inspection.setInspectionDate(LocalDateTime.now());
        inspection.setCreatedAt(LocalDateTime.now());
        inspection.setUpdatedAt(LocalDateTime.now());

        updateItemQuantities(order.getId(), inspection);

        order.setStatus(RmaOrderStatus.INSPECTED);
        order.setUpdatedAt(LocalDateTime.now());

        log.info("Recorded inspection for order: id={}, rmaNo={}, result={}", order.getId(), order.getRmaNo(), inspection.getResult());
        return inspection;
    }

    @Override
    public RmaDisposition makeDisposition(RmaOrder order, RmaDisposition disposition) {
        if (!order.canDispose()) {
            throw new IllegalStateException("RMA order cannot be disposed, current status: " + order.getStatus());
        }

        disposition.setRmaId(order.getId());
        disposition.setRmaNo(order.getRmaNo());
        disposition.setDispositionDate(LocalDateTime.now());
        disposition.setCreatedAt(LocalDateTime.now());
        disposition.setUpdatedAt(LocalDateTime.now());

        order.setStatus(RmaOrderStatus.AWAITING_DISPOSITION);
        order.setUpdatedAt(LocalDateTime.now());

        log.info("Made disposition for order: id={}, rmaNo={}, type={}", order.getId(), order.getRmaNo(), disposition.getDispositionType());
        return disposition;
    }

    @Override
    public RmaOrder executeDisposition(Long rmaId) {
        RmaOrder order = findOrderOrThrow(rmaId);
        if (RmaOrderStatus.AWAITING_DISPOSITION != order.getStatus()) {
            throw new IllegalStateException("RMA order has no pending disposition, current status: " + order.getStatus());
        }

        order.setStatus(RmaOrderStatus.DISPOSED);
        order.setUpdatedAt(LocalDateTime.now());

        log.info("Executed disposition for order: id={}, rmaNo={}", rmaId, order.getRmaNo());
        return order;
    }

    @Override
    public RmaOrder completeRmaOrder(Long rmaId) {
        RmaOrder order = findOrderOrThrow(rmaId);
        if (!order.canComplete()) {
            throw new IllegalStateException("RMA order cannot be completed, current status: " + order.getStatus());
        }
        order.setStatus(RmaOrderStatus.COMPLETED);
        order.setUpdatedAt(LocalDateTime.now());

        log.info("Completed RMA order: id={}, rmaNo={}", rmaId, order.getRmaNo());
        return order;
    }

    @Override
    public RmaOrder cancelRmaOrder(Long rmaId) {
        RmaOrder order = findOrderOrThrow(rmaId);
        if (!order.canCancel()) {
            throw new IllegalStateException("RMA order cannot be cancelled, current status: " + order.getStatus());
        }
        order.setStatus(RmaOrderStatus.CANCELLED);
        order.setUpdatedAt(LocalDateTime.now());

        log.info("Cancelled RMA order: id={}, rmaNo={}", rmaId, order.getRmaNo());
        return order;
    }

    @Override
    public Optional<RmaOrder> getRmaOrder(Long rmaId) {
        return rmaOrderRepository.findById(rmaId);
    }

    @Override
    public Optional<RmaOrder> getRmaOrderByNo(String rmaNo) {
        return rmaOrderRepository.findByRmaNo(rmaNo);
    }

    private RmaOrder buildOrder(String orderNo, Long customerId, String customerName,
                                String contactPhone, String reasonCode, String reasonDescription,
                                Long warehouseId, String returnType, String createdBy,
                                List<RmaOrderItem> items) {
        RmaOrder order = new RmaOrder();
        order.setRmaNo(rmaSequenceGenerator.generateRmaNo());
        assignOrderDefaults(order, orderNo, customerId, customerName, contactPhone,
                reasonCode, reasonDescription, warehouseId, returnType, createdBy, items);
        return order;
    }

    private void assignOrderDefaults(RmaOrder order, String orderNo, Long customerId, String customerName,
                                     String contactPhone, String reasonCode, String reasonDescription,
                                     Long warehouseId, String returnType, String createdBy,
                                     List<RmaOrderItem> items) {
        order.setOrderNo(orderNo);
        order.setReturnType(returnType);
        order.setStatus(RmaOrderStatus.PENDING);
        order.setCustomerId(customerId);
        order.setCustomerName(customerName);
        order.setContactPhone(contactPhone);
        order.setReasonCode(reasonCode);
        order.setReasonDescription(reasonDescription);
        order.setWarehouseId(warehouseId);
        order.setTotalQuantity(calcTotalQuantity(items));
        order.setTotalAmount(calcTotalAmount(items));
        order.setCurrency(CURRENCY_CNY);
        order.setCreatedBy(createdBy);
        order.setCreatedAt(LocalDateTime.now());
        order.setUpdatedAt(LocalDateTime.now());
        order.setVersion(INITIAL_VERSION);
    }

    private void updateItemQuantities(Long rmaId, RmaInspection inspection) {
        List<RmaOrderItem> items = rmaOrderItemRepository.findByRmaId(rmaId);
        if (items == null || items.isEmpty()) return;
        int totalExpected = items.stream().mapToInt(RmaOrderItem::getExpectedQuantity).sum();
        if (totalExpected == 0) return;
        for (RmaOrderItem item : items) {
            int proportion = item.getExpectedQuantity();
            item.setInspectedQuantity((int) Math.round(1.0 * inspection.getTotalInspected() * proportion / totalExpected));
            item.setAcceptedQuantity((int) Math.round(1.0 * inspection.getTotalPassed() * proportion / totalExpected));
            rmaOrderItemRepository.save(item);
        }
    }

    private int calcTotalQuantity(List<RmaOrderItem> items) {
        return items.stream().mapToInt(RmaOrderItem::getExpectedQuantity).sum();
    }

    private BigDecimal calcTotalAmount(List<RmaOrderItem> items) {
        return items.stream()
                .map(i -> i.getUnitPrice().multiply(BigDecimal.valueOf(i.getExpectedQuantity())))
                .reduce(BigDecimal.ZERO, BigDecimal::add);
    }

    private RmaOrder findOrderOrThrow(Long rmaId) {
        return rmaOrderRepository.findById(rmaId)
                .orElseThrow(() -> new IllegalArgumentException("RMA order not found: " + rmaId));
    }
}
