package com.metawebthree.routeoptimizer.domain.entity;

import java.time.LocalDateTime;

public class RoutePoint {
    private Long id;
    private String pointCode;
    private String pointName;
    private PointType type;
    private Double latitude;
    private Double longitude;
    private String address;
    private String contactPerson;
    private String contactPhone;
    private Integer sequence;
    private Double estimatedArrivalTime;
    private Double actualArrivalTime;
    private Integer expectedServiceDuration;
    private Integer actualServiceDuration;
    private PointStatus status;
    private Double distanceFromPrevious;
    private String orderCode;
    private String remarks;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;

    public enum PointType {
        WAREHOUSE, CUSTOMER, PICKUP, DROP_OFF, SERVICE, REST
    }

    public enum PointStatus {
        PENDING, ARRIVED, SERVING, COMPLETED, SKIPPED, FAILED
    }

    public RoutePoint() {
        this.status = PointStatus.PENDING;
        this.createdAt = LocalDateTime.now();
        this.updatedAt = LocalDateTime.now();
    }

    public double calculateDistanceTo(RoutePoint other) {
        if (other == null) return 0.0;
        return calculateHaversineDistance(
            this.latitude, this.longitude,
            other.latitude, other.longitude
        );
    }

    private double calculateHaversineDistance(double lat1, double lon1, double lat2, double lon2) {
        final double R = 6371.0;
        double latDistance = Math.toRadians(lat2 - lat1);
        double lonDistance = Math.toRadians(lon2 - lon1);
        double a = Math.sin(latDistance / 2) * Math.sin(latDistance / 2)
            + Math.cos(Math.toRadians(lat1)) * Math.cos(Math.toRadians(lat2))
            * Math.sin(lonDistance / 2) * Math.sin(lonDistance / 2);
        double c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
        return R * c;
    }

    public void arrive() {
        if (this.status != PointStatus.PENDING) {
            throw new IllegalStateException("Can only arrive at pending points");
        }
        this.status = PointStatus.ARRIVED;
        this.actualArrivalTime = 0.0;
        this.updatedAt = LocalDateTime.now();
    }

    public void completeService() {
        if (this.status != PointStatus.ARRIVED && this.status != PointStatus.SERVING) {
            throw new IllegalStateException("Can only complete arrived or serving points");
        }
        this.status = PointStatus.COMPLETED;
        this.updatedAt = LocalDateTime.now();
    }

    public void skip() {
        this.status = PointStatus.SKIPPED;
        this.updatedAt = LocalDateTime.now();
    }

    // Getters and Setters
    public Long getId() { return id; }
    public void setId(Long id) { this.id = id; }
    public String getPointCode() { return pointCode; }
    public void setPointCode(String pointCode) { this.pointCode = pointCode; }
    public String getPointName() { return pointName; }
    public void setPointName(String pointName) { this.pointName = pointName; }
    public PointType getType() { return type; }
    public void setType(PointType type) { this.type = type; }
    public Double getLatitude() { return latitude; }
    public void setLatitude(Double latitude) { this.latitude = latitude; }
    public Double getLongitude() { return longitude; }
    public void setLongitude(Double longitude) { this.longitude = longitude; }
    public String getAddress() { return address; }
    public void setAddress(String address) { this.address = address; }
    public String getContactPerson() { return contactPerson; }
    public void setContactPerson(String contactPerson) { this.contactPerson = contactPerson; }
    public String getContactPhone() { return contactPhone; }
    public void setContactPhone(String contactPhone) { this.contactPhone = contactPhone; }
    public Integer getSequence() { return sequence; }
    public void setSequence(Integer sequence) { this.sequence = sequence; }
    public Double getEstimatedArrivalTime() { return estimatedArrivalTime; }
    public void setEstimatedArrivalTime(Double estimatedArrivalTime) { this.estimatedArrivalTime = estimatedArrivalTime; }
    public Double getActualArrivalTime() { return actualArrivalTime; }
    public void setActualArrivalTime(Double actualArrivalTime) { this.actualArrivalTime = actualArrivalTime; }
    public Integer getExpectedServiceDuration() { return expectedServiceDuration; }
    public void setExpectedServiceDuration(Integer expectedServiceDuration) { this.expectedServiceDuration = expectedServiceDuration; }
    public Integer getActualServiceDuration() { return actualServiceDuration; }
    public void setActualServiceDuration(Integer actualServiceDuration) { this.actualServiceDuration = actualServiceDuration; }
    public PointStatus getStatus() { return status; }
    public void setStatus(PointStatus status) { this.status = status; }
    public Double getDistanceFromPrevious() { return distanceFromPrevious; }
    public void setDistanceFromPrevious(Double distanceFromPrevious) { this.distanceFromPrevious = distanceFromPrevious; }
    public String getOrderCode() { return orderCode; }
    public void setOrderCode(String orderCode) { this.orderCode = orderCode; }
    public String getRemarks() { return remarks; }
    public void setRemarks(String remarks) { this.remarks = remarks; }
    public LocalDateTime getCreatedAt() { return createdAt; }
    public void setCreatedAt(LocalDateTime createdAt) { this.createdAt = createdAt; }
    public LocalDateTime getUpdatedAt() { return updatedAt; }
    public void setUpdatedAt(LocalDateTime updatedAt) { this.updatedAt = updatedAt; }
}