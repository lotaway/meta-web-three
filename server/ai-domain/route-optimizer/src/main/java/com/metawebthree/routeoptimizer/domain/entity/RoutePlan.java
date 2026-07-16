package com.metawebthree.routeoptimizer.domain.entity;

import jakarta.persistence.*;
import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.List;

@Entity
@Table(name = "tb_route_plan")
public class RoutePlan {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Column(name = "plan_code", unique = true)
    private String planCode;

    @Column(name = "plan_name")
    private String planName;

    @Column(name = "vehicle_code")
    private String vehicleCode;

    @Column(name = "driver_name")
    private String driverName;

    @Column(name = "driver_phone")
    private String driverPhone;

    @Enumerated(EnumType.STRING)
    @Column(name = "status")
    private RouteStatus status;

    @Column(name = "total_distance")
    private Double totalDistance;

    @Column(name = "estimated_duration")
    private Integer estimatedDuration;

    @Column(name = "planned_start_time")
    private LocalDateTime plannedStartTime;

    @Column(name = "planned_end_time")
    private LocalDateTime plannedEndTime;

    @Column(name = "actual_start_time")
    private LocalDateTime actualStartTime;

    @Column(name = "actual_end_time")
    private LocalDateTime actualEndTime;

    @Enumerated(EnumType.STRING)
    @Column(name = "optimization_type")
    private OptimizationType optimizationType;

    @Column(name = "total_cost")
    private Double totalCost;

    @Column(name = "remarks")
    private String remarks;

    @Column(name = "created_at")
    private LocalDateTime createdAt;

    @Column(name = "updated_at")
    private LocalDateTime updatedAt;

    @OneToMany(cascade = CascadeType.ALL, orphanRemoval = true)
    @JoinColumn(name = "route_plan_id")
    private List<RoutePoint> points;

    public enum RouteStatus {
        PENDING, OPTIMIZING, OPTIMIZED, IN_PROGRESS, COMPLETED, CANCELLED
    }

    public enum OptimizationType {
        DISTANCE_MINIMIZE, TIME_MINIMIZE, COST_MINIMIZE, BALANCED
    }

    public RoutePlan() {
        this.points = new ArrayList<>();
        this.status = RouteStatus.PENDING;
        this.createdAt = LocalDateTime.now();
        this.updatedAt = LocalDateTime.now();
    }

    public void addPoint(RoutePoint point) {
        this.points.add(point);
        point.setSequence(this.points.size());
    }

    public void calculateTotalDistance() {
        if (this.points.size() < 2) {
            this.totalDistance = 0.0;
            return;
        }
        double total = 0.0;
        for (int i = 0; i < this.points.size() - 1; i++) {
            total += this.points.get(i).calculateDistanceTo(this.points.get(i + 1));
        }
        this.totalDistance = total;
    }

    public void startExecution() {
        if (this.status != RouteStatus.OPTIMIZED) {
            throw new IllegalStateException("Can only start optimized routes");
        }
        this.status = RouteStatus.IN_PROGRESS;
        this.actualStartTime = LocalDateTime.now();
        this.updatedAt = LocalDateTime.now();
    }

    public void completeRoute() {
        if (this.status != RouteStatus.IN_PROGRESS) {
            throw new IllegalStateException("Can only complete routes in progress");
        }
        this.status = RouteStatus.COMPLETED;
        this.actualEndTime = LocalDateTime.now();
        this.updatedAt = LocalDateTime.now();
    }

    // Getters and Setters
    public Long getId() { return id; }
    public void setId(Long id) { this.id = id; }
    public String getPlanCode() { return planCode; }
    public void setPlanCode(String planCode) { this.planCode = planCode; }
    public String getPlanName() { return planName; }
    public void setPlanName(String planName) { this.planName = planName; }
    public String getVehicleCode() { return vehicleCode; }
    public void setVehicleCode(String vehicleCode) { this.vehicleCode = vehicleCode; }
    public String getDriverName() { return driverName; }
    public void setDriverName(String driverName) { this.driverName = driverName; }
    public String getDriverPhone() { return driverPhone; }
    public void setDriverPhone(String driverPhone) { this.driverPhone = driverPhone; }
    public RouteStatus getStatus() { return status; }
    public void setStatus(RouteStatus status) { this.status = status; }
    public Double getTotalDistance() { return totalDistance; }
    public void setTotalDistance(Double totalDistance) { this.totalDistance = totalDistance; }
    public Integer getEstimatedDuration() { return estimatedDuration; }
    public void setEstimatedDuration(Integer estimatedDuration) { this.estimatedDuration = estimatedDuration; }
    public LocalDateTime getPlannedStartTime() { return plannedStartTime; }
    public void setPlannedStartTime(LocalDateTime plannedStartTime) { this.plannedStartTime = plannedStartTime; }
    public LocalDateTime getPlannedEndTime() { return plannedEndTime; }
    public void setPlannedEndTime(LocalDateTime plannedEndTime) { this.plannedEndTime = plannedEndTime; }
    public LocalDateTime getActualStartTime() { return actualStartTime; }
    public void setActualStartTime(LocalDateTime actualStartTime) { this.actualStartTime = actualStartTime; }
    public LocalDateTime getActualEndTime() { return actualEndTime; }
    public void setActualEndTime(LocalDateTime actualEndTime) { this.actualEndTime = actualEndTime; }
    public OptimizationType getOptimizationType() { return optimizationType; }
    public void setOptimizationType(OptimizationType optimizationType) { this.optimizationType = optimizationType; }
    public Double getTotalCost() { return totalCost; }
    public void setTotalCost(Double totalCost) { this.totalCost = totalCost; }
    public String getRemarks() { return remarks; }
    public void setRemarks(String remarks) { this.remarks = remarks; }
    public LocalDateTime getCreatedAt() { return createdAt; }
    public void setCreatedAt(LocalDateTime createdAt) { this.createdAt = createdAt; }
    public LocalDateTime getUpdatedAt() { return updatedAt; }
    public void setUpdatedAt(LocalDateTime updatedAt) { this.updatedAt = updatedAt; }
    public List<RoutePoint> getPoints() { return points; }
    public void setPoints(List<RoutePoint> points) { this.points = points; }
}