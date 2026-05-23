package com.metawebthree.routeoptimizer.domain.entity;

import java.time.LocalDateTime;

public class Vehicle {
    private Long id;
    private String vehicleCode;
    private String vehicleNumber;
    private String vehicleType;
    private VehicleStatus status;
    private Double maxLoadCapacity;
    private Double currentLoad;
    private Double fuelCapacity;
    private Double currentFuel;
    private Double fuelEfficiency;
    private String driverName;
    private String driverPhone;
    private Double latitude;
    private Double longitude;
    private LocalDateTime lastLocationUpdate;
    private String currentRoutePlanCode;
    private Integer totalDeliveries;
    private Double totalDistance;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;

    public enum VehicleStatus {
        IDLE, ASSIGNED, IN_TRANSIT, MAINTENANCE, OFFLINE
    }

    public Vehicle() {
        this.status = VehicleStatus.IDLE;
        this.currentLoad = 0.0;
        this.currentFuel = 0.0;
        this.totalDeliveries = 0;
        this.totalDistance = 0.0;
        this.createdAt = LocalDateTime.now();
        this.updatedAt = LocalDateTime.now();
    }

    public boolean canAcceptLoad(Double weight) {
        return this.currentLoad + weight <= this.maxLoadCapacity;
    }

    public void assignToRoute(String routePlanCode) {
        if (this.status != VehicleStatus.IDLE) {
            throw new IllegalStateException("Can only assign idle vehicles");
        }
        this.status = VehicleStatus.ASSIGNED;
        this.currentRoutePlanCode = routePlanCode;
        this.updatedAt = LocalDateTime.now();
    }

    public void startDelivery() {
        if (this.status != VehicleStatus.ASSIGNED) {
            throw new IllegalStateException("Can only start assigned vehicles");
        }
        this.status = VehicleStatus.IN_TRANSIT;
        this.updatedAt = LocalDateTime.now();
    }

    public void completeDelivery() {
        if (this.status != VehicleStatus.IN_TRANSIT) {
            throw new IllegalStateException("Can only complete in-transit vehicles");
        }
        this.status = VehicleStatus.IDLE;
        this.currentRoutePlanCode = null;
        this.totalDeliveries++;
        this.updatedAt = LocalDateTime.now();
    }

    public void updateLocation(Double latitude, Double longitude) {
        this.latitude = latitude;
        this.longitude = longitude;
        this.lastLocationUpdate = LocalDateTime.now();
        this.updatedAt = LocalDateTime.now();
    }

    // Getters and Setters
    public Long getId() { return id; }
    public void setId(Long id) { this.id = id; }
    public String getVehicleCode() { return vehicleCode; }
    public void setVehicleCode(String vehicleCode) { this.vehicleCode = vehicleCode; }
    public String getVehicleNumber() { return vehicleNumber; }
    public void setVehicleNumber(String vehicleNumber) { this.vehicleNumber = vehicleNumber; }
    public String getVehicleType() { return vehicleType; }
    public void setVehicleType(String vehicleType) { this.vehicleType = vehicleType; }
    public VehicleStatus getStatus() { return status; }
    public void setStatus(VehicleStatus status) { this.status = status; }
    public Double getMaxLoadCapacity() { return maxLoadCapacity; }
    public void setMaxLoadCapacity(Double maxLoadCapacity) { this.maxLoadCapacity = maxLoadCapacity; }
    public Double getCurrentLoad() { return currentLoad; }
    public void setCurrentLoad(Double currentLoad) { this.currentLoad = currentLoad; }
    public Double getFuelCapacity() { return fuelCapacity; }
    public void setFuelCapacity(Double fuelCapacity) { this.fuelCapacity = fuelCapacity; }
    public Double getCurrentFuel() { return currentFuel; }
    public void setCurrentFuel(Double currentFuel) { this.currentFuel = currentFuel; }
    public Double getFuelEfficiency() { return fuelEfficiency; }
    public void setFuelEfficiency(Double fuelEfficiency) { this.fuelEfficiency = fuelEfficiency; }
    public String getDriverName() { return driverName; }
    public void setDriverName(String driverName) { this.driverName = driverName; }
    public String getDriverPhone() { return driverPhone; }
    public void setDriverPhone(String driverPhone) { this.driverPhone = driverPhone; }
    public Double getLatitude() { return latitude; }
    public void setLatitude(Double latitude) { this.latitude = latitude; }
    public Double getLongitude() { return longitude; }
    public void setLongitude(Double longitude) { this.longitude = longitude; }
    public LocalDateTime getLastLocationUpdate() { return lastLocationUpdate; }
    public void setLastLocationUpdate(LocalDateTime lastLocationUpdate) { this.lastLocationUpdate = lastLocationUpdate; }
    public String getCurrentRoutePlanCode() { return currentRoutePlanCode; }
    public void setCurrentRoutePlanCode(String currentRoutePlanCode) { this.currentRoutePlanCode = currentRoutePlanCode; }
    public Integer getTotalDeliveries() { return totalDeliveries; }
    public void setTotalDeliveries(Integer totalDeliveries) { this.totalDeliveries = totalDeliveries; }
    public Double getTotalDistance() { return totalDistance; }
    public void setTotalDistance(Double totalDistance) { this.totalDistance = totalDistance; }
    public LocalDateTime getCreatedAt() { return createdAt; }
    public void setCreatedAt(LocalDateTime createdAt) { this.createdAt = createdAt; }
    public LocalDateTime getUpdatedAt() { return updatedAt; }
    public void setUpdatedAt(LocalDateTime updatedAt) { this.updatedAt = updatedAt; }
}