package com.metawebthree.dom.domain.service;

import jakarta.validation.constraints.NotEmpty;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import jakarta.annotation.PostConstruct;
import java.util.Arrays;
import java.util.List;

public class DomSourcingProperties {

    private static final Logger log = LoggerFactory.getLogger(DomSourcingProperties.class);

    @NotEmpty(message = "At least one warehouse ID must be configured")
    private List<Long> warehouseIds = Arrays.asList(1L, 2L, 3L);
    private double shippingCostPerKm = 0.5;
    private double handlingCostFlat = 10.0;
    private double distanceScoreFactor = 100.0;
    private double costScoreFactor = 1000.0;
    private double balancedDistanceWeight = 0.5;
    private double balancedCostWeight = 0.5;
    private String whNamePrefix = "WH-";

    public List<Long> getWarehouseIds() { return warehouseIds; }
    public void setWarehouseIds(List<Long> warehouseIds) { this.warehouseIds = warehouseIds; }

    public double getShippingCostPerKm() { return shippingCostPerKm; }
    public void setShippingCostPerKm(double shippingCostPerKm) { this.shippingCostPerKm = shippingCostPerKm; }

    public double getHandlingCostFlat() { return handlingCostFlat; }
    public void setHandlingCostFlat(double handlingCostFlat) { this.handlingCostFlat = handlingCostFlat; }

    public double getDistanceScoreFactor() { return distanceScoreFactor; }
    public void setDistanceScoreFactor(double distanceScoreFactor) { this.distanceScoreFactor = distanceScoreFactor; }

    public double getCostScoreFactor() { return costScoreFactor; }
    public void setCostScoreFactor(double costScoreFactor) { this.costScoreFactor = costScoreFactor; }

    public double getBalancedDistanceWeight() { return balancedDistanceWeight; }
    public void setBalancedDistanceWeight(double balancedDistanceWeight) { this.balancedDistanceWeight = balancedDistanceWeight; }

    public double getBalancedCostWeight() { return balancedCostWeight; }
    public void setBalancedCostWeight(double balancedCostWeight) { this.balancedCostWeight = balancedCostWeight; }

    public String getWhNamePrefix() { return whNamePrefix; }
    public void setWhNamePrefix(String whNamePrefix) { this.whNamePrefix = whNamePrefix; }

    @PostConstruct
    public void validate() {
        if (warehouseIds == null || warehouseIds.isEmpty()) {
            log.warn("No warehouse IDs configured via dom.sourcing.warehouse-ids, using defaults [1L, 2L, 3L]");
            this.warehouseIds = Arrays.asList(1L, 2L, 3L);
        }
    }
}
