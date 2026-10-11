package com.metawebthree.inventoryalert.application.service;

import com.metawebthree.inventoryalert.application.dto.InventoryAlertDTO;
import com.metawebthree.inventoryalert.domain.model.AlertLevel;
import com.metawebthree.inventoryalert.domain.model.AlertStatus;
import com.metawebthree.inventoryalert.domain.model.InventoryAlertDO;
import com.metawebthree.inventoryalert.domain.repository.InventoryAlertRepository;
import com.metawebthree.inventoryalert.infrastructure.client.InventoryClient;
import com.metawebthree.common.exception.BusinessException;
import com.metawebthree.common.enums.ResponseStatus;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.List;
import java.util.stream.Collectors;

@Service
public class InventoryAlertApplicationService {

    private final InventoryAlertRepository alertRepository;
    private final InventoryClient inventoryClient;

    public InventoryAlertApplicationService(InventoryAlertRepository alertRepository,
                                              InventoryClient inventoryClient) {
        this.alertRepository = alertRepository;
        this.inventoryClient = inventoryClient;
    }

    /**
     * Check inventory levels and create alerts if needed
     */
    @Transactional
    public void checkInventory(Long productId, Integer threshold) {
        boolean isBelow = inventoryClient.isBelowThreshold(productId, threshold);
        
        if (isBelow) {
            // Check if alert already exists
            List<InventoryAlertDO> existingAlerts = alertRepository.findByProductId(productId);
            boolean hasPendingAlert = existingAlerts.stream()
                    .anyMatch(a -> a.getAlertStatus().equals(AlertStatus.PENDING.getCode()) 
                                || a.getAlertStatus().equals(AlertStatus.NOTIFIED.getCode()));
            
            if (!hasPendingAlert) {
                // Get current inventory
                var inventory = inventoryClient.getInventory(productId, "DEFAULT");
                int currentStock = inventory != null ? inventory.getAvailableQty() : 0;

                // Determine alert level based on stock percentage
                int alertLevel = determineAlertLevel(currentStock, threshold);
                
                InventoryAlertDO alert = new InventoryAlertDO();
                alert.setProductId(productId);
                alert.setCurrentStock(currentStock);
                alert.setThreshold(threshold);
                alert.setAlertLevel(alertLevel);
                alert.setAlertStatus(AlertStatus.PENDING.getCode());
                alert.setAlertMessage(String.format("Stock below threshold: %d < %d", currentStock, threshold));
                alert.setAlertTime(LocalDateTime.now());

                alertRepository.save(alert);
            }
        }
    }

    /**
     * Process alert (resolve or ignore)
     */
    @Transactional
    public InventoryAlertDTO processAlert(Long alertId, boolean resolved, String remark) {
        InventoryAlertDO alert = alertRepository.findById(alertId);
        if (alert == null) {
            throw new BusinessException(ResponseStatus.NOT_FOUND, "Alert not found");
        }

        if (resolved) {
            alert.setAlertStatus(AlertStatus.RESOLVED.getCode());
            alert.setResolvedTime(LocalDateTime.now());
        } else {
            alert.setAlertStatus(AlertStatus.IGNORED.getCode());
        }

        if (remark != null) {
            alert.setRemark(remark);
        }

        alertRepository.save(alert);
        return convertToDTO(alert);
    }

    /**
     * Get all alerts (admin)
     */
    public List<InventoryAlertDTO> getAll() {
        return alertRepository.findAll().stream()
                .map(this::convertToDTO)
                .collect(Collectors.toList());
    }

    /**
     * Get alerts by status
     */
    public List<InventoryAlertDTO> getByStatus(Integer status) {
        return alertRepository.findByStatus(status).stream()
                .map(this::convertToDTO)
                .collect(Collectors.toList());
    }

    /**
     * Get alerts by alert level
     */
    public List<InventoryAlertDTO> getByAlertLevel(Integer alertLevel) {
        return alertRepository.findByAlertLevel(alertLevel).stream()
                .map(this::convertToDTO)
                .collect(Collectors.toList());
    }

    /**
     * Get alerts by product ID
     */
    public List<InventoryAlertDTO> getByProductId(Long productId) {
        return alertRepository.findByProductId(productId).stream()
                .map(this::convertToDTO)
                .collect(Collectors.toList());
    }

    /**
     * Get high priority alerts
     */
    public List<InventoryAlertDTO> getHighPriorityAlerts() {
        return alertRepository.findAll().stream()
                .filter(a -> a.getAlertLevel() >= AlertLevel.HIGH.getCode())
                .filter(a -> a.getAlertStatus() <= AlertStatus.NOTIFIED.getCode())
                .map(this::convertToDTO)
                .collect(Collectors.toList());
    }

    /**
     * Analyze inventory turnover
     */
    public InventoryAlertDTO analyzeTurnover(Long productId) {
        InventoryAlertDO alert = new InventoryAlertDO();
        alert.setProductId(productId);
        
        // Simplified turnover analysis
        var inventory = inventoryClient.getInventory(productId, "DEFAULT");
        if (inventory != null) {
            int totalQty = inventory.getTotalQty();
            alert.setCurrentStock(totalQty);
            
            // If stock is 0, it's a critical alert
            if (totalQty == 0) {
                alert.setAlertLevel(AlertLevel.CRITICAL.getCode());
                alert.setAlertMessage("Product out of stock");
            }
        }
        
        return convertToDTO(alert);
    }

    private int determineAlertLevel(int currentStock, int threshold) {
        double ratio = (double) currentStock / threshold;
        if (ratio <= 0.25) {
            return AlertLevel.CRITICAL.getCode();
        } else if (ratio <= 0.5) {
            return AlertLevel.HIGH.getCode();
        } else if (ratio <= 0.75) {
            return AlertLevel.MEDIUM.getCode();
        } else {
            return AlertLevel.LOW.getCode();
        }
    }

    private InventoryAlertDTO convertToDTO(InventoryAlertDO alert) {
        InventoryAlertDTO dto = new InventoryAlertDTO();
        dto.setId(alert.getId());
        dto.setProductId(alert.getProductId());
        dto.setSkuId(alert.getSkuId());
        dto.setProductName(alert.getProductName());
        dto.setSkuCode(alert.getSkuCode());
        dto.setWarehouseId(alert.getWarehouseId());
        dto.setWarehouseName(alert.getWarehouseName());
        dto.setCurrentStock(alert.getCurrentStock());
        dto.setThreshold(alert.getThreshold());
        dto.setAlertLevel(alert.getAlertLevel());
        dto.setAlertStatus(alert.getAlertStatus());
        dto.setAlertMessage(alert.getAlertMessage());
        dto.setRemark(alert.getRemark());

        // Set alert level description
        if (alert.getAlertLevel() != null) {
            for (AlertLevel level : AlertLevel.values()) {
                if (level.getCode().equals(alert.getAlertLevel())) {
                    dto.setAlertLevelDesc(level.getDesc());
                    break;
                }
            }
        }

        // Set alert status description
        if (alert.getAlertStatus() != null) {
            for (AlertStatus status : AlertStatus.values()) {
                if (status.getCode().equals(alert.getAlertStatus())) {
                    dto.setAlertStatusDesc(status.getDesc());
                    break;
                }
            }
        }

        // Format times
        if (alert.getAlertTime() != null) {
            dto.setAlertTime(alert.getAlertTime().format(DateTimeFormatter.ISO_LOCAL_DATE_TIME));
        }
        if (alert.getResolvedTime() != null) {
            dto.setResolvedTime(alert.getResolvedTime().format(DateTimeFormatter.ISO_LOCAL_DATE_TIME));
        }

        return dto;
    }
}