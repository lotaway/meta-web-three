package com.metawebthree.routeoptimizer.infrastructure.persistence.repository;

import com.metawebthree.routeoptimizer.domain.entity.Vehicle;
import com.metawebthree.routeoptimizer.domain.repository.VehicleRepository;
import org.springframework.stereotype.Repository;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.Collectors;

@Repository
public class VehicleRepositoryImpl implements VehicleRepository {
    private final Map<Long, Vehicle> storage = new ConcurrentHashMap<>();
    private final Map<String, Vehicle> codeIndex = new ConcurrentHashMap<>();
    private final Map<String, Vehicle> numberIndex = new ConcurrentHashMap<>();
    private Long idGenerator = 1L;

    @Override
    public Optional<Vehicle> findById(Long id) {
        return Optional.ofNullable(storage.get(id));
    }

    @Override
    public Optional<Vehicle> findByVehicleCode(String vehicleCode) {
        return Optional.ofNullable(codeIndex.get(vehicleCode));
    }

    @Override
    public Optional<Vehicle> findByVehicleNumber(String vehicleNumber) {
        return Optional.ofNullable(numberIndex.get(vehicleNumber));
    }

    @Override
    public List<Vehicle> findByStatus(Vehicle.VehicleStatus status) {
        return storage.values().stream()
            .filter(v -> v.getStatus() == status)
            .collect(Collectors.toList());
    }

    @Override
    public List<Vehicle> findAll() {
        return new ArrayList<>(storage.values());
    }

    @Override
    public Vehicle save(Vehicle vehicle) {
        if (vehicle.getId() == null) {
            vehicle.setId(idGenerator++);
        }
        storage.put(vehicle.getId(), vehicle);
        if (vehicle.getVehicleCode() != null) {
            codeIndex.put(vehicle.getVehicleCode(), vehicle);
        }
        if (vehicle.getVehicleNumber() != null) {
            numberIndex.put(vehicle.getVehicleNumber(), vehicle);
        }
        return vehicle;
    }

    @Override
    public void delete(Vehicle vehicle) {
        if (vehicle.getId() != null) {
            storage.remove(vehicle.getId());
        }
        if (vehicle.getVehicleCode() != null) {
            codeIndex.remove(vehicle.getVehicleCode());
        }
        if (vehicle.getVehicleNumber() != null) {
            numberIndex.remove(vehicle.getVehicleNumber());
        }
    }

    @Override
    public List<Vehicle> findAvailableVehicles() {
        return storage.values().stream()
            .filter(v -> v.getStatus() == Vehicle.VehicleStatus.IDLE)
            .collect(Collectors.toList());
    }
}