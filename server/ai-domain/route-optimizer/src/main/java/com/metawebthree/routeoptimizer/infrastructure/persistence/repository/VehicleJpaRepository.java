package com.metawebthree.routeoptimizer.infrastructure.persistence.repository;

import com.metawebthree.routeoptimizer.domain.entity.Vehicle;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import java.util.List;
import java.util.Optional;

@Repository
public interface VehicleJpaRepository extends JpaRepository<Vehicle, Long> {
    Optional<Vehicle> findByVehicleCode(String vehicleCode);
    Optional<Vehicle> findByVehicleNumber(String vehicleNumber);
    List<Vehicle> findByStatus(Vehicle.VehicleStatus status);
}
