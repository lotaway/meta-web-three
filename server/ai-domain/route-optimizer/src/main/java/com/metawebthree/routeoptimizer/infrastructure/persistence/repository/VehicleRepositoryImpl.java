package com.metawebthree.routeoptimizer.infrastructure.persistence.repository;

import com.metawebthree.routeoptimizer.domain.entity.Vehicle;
import com.metawebthree.routeoptimizer.domain.repository.VehicleRepository;
import org.springframework.stereotype.Repository;

import java.util.List;
import java.util.Optional;

@Repository
public class VehicleRepositoryImpl implements VehicleRepository {
    private final VehicleJpaRepository repository;

    public VehicleRepositoryImpl(VehicleJpaRepository repository) {
        this.repository = repository;
    }

    @Override
    public Optional<Vehicle> findById(Long id) {
        return repository.findById(id);
    }

    @Override
    public Optional<Vehicle> findByVehicleCode(String vehicleCode) {
        return repository.findByVehicleCode(vehicleCode);
    }

    @Override
    public Optional<Vehicle> findByVehicleNumber(String vehicleNumber) {
        return repository.findByVehicleNumber(vehicleNumber);
    }

    @Override
    public List<Vehicle> findByStatus(Vehicle.VehicleStatus status) {
        return repository.findByStatus(status);
    }

    @Override
    public List<Vehicle> findAll() {
        return repository.findAll();
    }

    @Override
    public Vehicle save(Vehicle vehicle) {
        return repository.save(vehicle);
    }

    @Override
    public void delete(Vehicle vehicle) {
        repository.delete(vehicle);
    }

    @Override
    public List<Vehicle> findAvailableVehicles() {
        return repository.findByStatus(Vehicle.VehicleStatus.IDLE);
    }
}
