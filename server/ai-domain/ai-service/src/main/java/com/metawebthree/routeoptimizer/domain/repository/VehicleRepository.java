package com.metawebthree.routeoptimizer.domain.repository;

import com.metawebthree.routeoptimizer.domain.entity.Vehicle;
import java.util.List;
import java.util.Optional;

public interface VehicleRepository {
    Optional<Vehicle> findById(Long id);
    Optional<Vehicle> findByVehicleCode(String vehicleCode);
    Optional<Vehicle> findByVehicleNumber(String vehicleNumber);
    List<Vehicle> findByStatus(Vehicle.VehicleStatus status);
    List<Vehicle> findAll();
    void save(Vehicle vehicle);
    void delete(Vehicle vehicle);
    List<Vehicle> findAvailableVehicles();
}