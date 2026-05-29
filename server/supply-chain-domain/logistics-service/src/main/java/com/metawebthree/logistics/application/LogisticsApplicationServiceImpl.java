package com.metawebthree.logistics.application;

import com.metawebthree.logistics.application.dto.LogisticsOrderDTO;
import com.metawebthree.logistics.domain.entity.LogisticsOrder;
import com.metawebthree.logistics.domain.entity.TrackingEvent;
import com.metawebthree.logistics.infrastructure.event.LogisticsDomainEventPublisher;
import com.metawebthree.logistics.infrastructure.persistence.repository.LogisticsOrderRepository;
import org.springframework.stereotype.Service;
import java.time.LocalDateTime;
import java.util.List;
import java.util.UUID;
import java.util.stream.Collectors;

@Service
public class LogisticsApplicationServiceImpl implements LogisticsApplicationService {

    private final LogisticsOrderRepository repository;
    private final LogisticsDomainEventPublisher eventPublisher;

    public LogisticsApplicationServiceImpl(LogisticsOrderRepository repository,
                                           LogisticsDomainEventPublisher eventPublisher) {
        this.repository = repository;
        this.eventPublisher = eventPublisher;
    }

    @Override
    public LogisticsOrderDTO createOrder(LogisticsOrderDTO dto) {
        LogisticsOrder order = new LogisticsOrder();
        String trackingNo = generateTrackingNo();
        order.setTrackingNo(trackingNo);
        order.setOrderNo(dto.getOrderNo());
        order.setCarrierId(dto.getCarrierId());
        order.setCarrierName(dto.getCarrierName());
        order.setServiceType(dto.getServiceType());
        order.setSenderName(dto.getSenderName());
        order.setSenderPhone(dto.getSenderPhone());
        order.setSenderProvince(dto.getSenderProvince());
        order.setSenderCity(dto.getSenderCity());
        order.setSenderDistrict(dto.getSenderDistrict());
        order.setSenderAddress(dto.getSenderAddress());
        order.setReceiverName(dto.getReceiverName());
        order.setReceiverPhone(dto.getReceiverPhone());
        order.setReceiverProvince(dto.getReceiverProvince());
        order.setReceiverCity(dto.getReceiverCity());
        order.setReceiverDistrict(dto.getReceiverDistrict());
        order.setReceiverAddress(dto.getReceiverAddress());
        order.setWeight(dto.getWeight());
        order.setVolume(dto.getVolume());
        order.setFreight(dto.getFreight());
        order.setStatus("CREATED");
        order.setCreatedAt(LocalDateTime.now());
        order.setUpdatedAt(LocalDateTime.now());
        
        LogisticsOrder saved = repository.save(order);
        eventPublisher.publishCreated(saved.getTrackingNo(), saved.getOrderNo());
        
        return toDTO(saved);
    }

    @Override
    public LogisticsOrderDTO queryByTrackingNo(String trackingNo) {
        return repository.findByTrackingNo(trackingNo)
            .map(this::toDTO)
            .orElse(null);
    }

    @Override
    public LogisticsOrderDTO queryByOrderNo(String orderNo) {
        return repository.findByOrderNo(orderNo)
            .map(this::toDTO)
            .orElse(null);
    }

    @Override
    public LogisticsOrderDTO updateStatus(String trackingNo, String status) {
        return repository.findByTrackingNo(trackingNo)
            .map(order -> {
                switch (status.toUpperCase()) {
                    case "PICKED_UP":
                        order.pickUp();
                        break;
                    case "IN_TRANSIT":
                        order.inTransit();
                        break;
                    case "OUT_FOR_DELIVERY":
                        order.outForDelivery();
                        break;
                    case "DELIVERED":
                        order.delivered();
                        eventPublisher.publishDelivered(trackingNo);
                        break;
                    case "EXCEPTION":
                        order.exception(null);
                        break;
                    default:
                        throw new IllegalArgumentException("Unknown status: " + status);
                }
                order.setUpdatedAt(LocalDateTime.now());
                LogisticsOrder updated = repository.save(order);
                eventPublisher.publishTrackingUpdated(trackingNo, status, null);
                return toDTO(updated);
            })
            .orElse(null);
    }

    @Override
    public List<LogisticsOrderDTO> listOrders(Long carrierId, String status) {
        List<LogisticsOrder> orders;
        if (carrierId != null && status != null) {
            orders = repository.findByCarrierIdAndStatus(carrierId, status);
        } else if (carrierId != null) {
            orders = repository.findByCarrierId(carrierId);
        } else if (status != null) {
            orders = repository.findByStatus(status);
        } else {
            orders = repository.findAll();
        }
        return orders.stream()
            .map(this::toDTO)
            .collect(Collectors.toList());
    }

    public LogisticsOrderDTO addTrackingEvent(String trackingNo, String eventType, String location, String description) {
        return repository.findByTrackingNo(trackingNo)
            .map(order -> {
                order.setUpdatedAt(LocalDateTime.now());
                LogisticsOrder updated = repository.save(order);
                eventPublisher.publishTrackingUpdated(trackingNo, eventType, location);
                return toDTO(updated);
            })
            .orElse(null);
    }

    private String generateTrackingNo() {
        return "LT" + System.currentTimeMillis() + UUID.randomUUID().toString().substring(0, 6).toUpperCase();
    }

    private LogisticsOrderDTO toDTO(LogisticsOrder order) {
        LogisticsOrderDTO dto = new LogisticsOrderDTO();
        dto.setId(order.getId());
        dto.setTrackingNo(order.getTrackingNo());
        dto.setOrderNo(order.getOrderNo());
        dto.setCarrierId(order.getCarrierId());
        dto.setCarrierName(order.getCarrierName());
        dto.setServiceType(order.getServiceType());
        dto.setSenderName(order.getSenderName());
        dto.setSenderPhone(order.getSenderPhone());
        dto.setSenderAddress(order.getSenderAddress());
        dto.setReceiverName(order.getReceiverName());
        dto.setReceiverPhone(order.getReceiverPhone());
        dto.setReceiverAddress(order.getReceiverAddress());
        dto.setWeight(order.getWeight());
        dto.setVolume(order.getVolume());
        dto.setFreight(order.getFreight());
        dto.setStatus(order.getStatus());
        dto.setPickedUpAt(order.getPickedUpAt());
        dto.setDeliveredAt(order.getDeliveredAt());
        dto.setCreatedAt(order.getCreatedAt());
        return dto;
    }
}