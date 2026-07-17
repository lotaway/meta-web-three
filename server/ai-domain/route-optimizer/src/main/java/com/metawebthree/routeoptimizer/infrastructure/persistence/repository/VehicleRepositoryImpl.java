package com.metawebthree.routeoptimizer.infrastructure.persistence.repository;

import com.baomidou.mybatisplus.core.conditions.query.QueryWrapper;
import com.metawebthree.routeoptimizer.domain.entity.Vehicle;
import com.metawebthree.routeoptimizer.domain.repository.VehicleRepository;
import com.metawebthree.routeoptimizer.infrastructure.persistence.mapper.VehicleMapper;
import org.springframework.stereotype.Repository;
import java.util.List;
import java.util.Optional;

@Repository
public class VehicleRepositoryImpl implements VehicleRepository {
    private final VehicleMapper mapper;

    public VehicleRepositoryImpl(VehicleMapper mapper) {
        this.mapper = mapper;
    }

    @Override
    public Optional<Vehicle> findById(Long id) {
        return Optional.ofNullable(mapper.selectById(id));
    }

    @Override
    public Optional<Vehicle> findByVehicleCode(String vehicleCode) {
        return Optional.ofNullable(
            mapper.selectOne(new QueryWrapper<Vehicle>().eq("vehicle_code", vehicleCode)));
    }

    @Override
    public Optional<Vehicle> findByVehicleNumber(String vehicleNumber) {
        return Optional.ofNullable(
            mapper.selectOne(new QueryWrapper<Vehicle>().eq("vehicle_number", vehicleNumber)));
    }

    @Override
    public List<Vehicle> findByStatus(Vehicle.VehicleStatus status) {
        return mapper.selectList(new QueryWrapper<Vehicle>().eq("status", status));
    }

    @Override
    public List<Vehicle> findAll() {
        return mapper.selectList(null);
    }

    @Override
    public void save(Vehicle vehicle) {
        if (vehicle.getId() != null) {
            mapper.updateById(vehicle);
        } else {
            mapper.insert(vehicle);
        }
    }

    @Override
    public void delete(Vehicle vehicle) {
        mapper.deleteById(vehicle.getId());
    }

    @Override
    public List<Vehicle> findAvailableVehicles() {
        return mapper.selectList(new QueryWrapper<Vehicle>().eq("status", Vehicle.VehicleStatus.IDLE));
    }
}
