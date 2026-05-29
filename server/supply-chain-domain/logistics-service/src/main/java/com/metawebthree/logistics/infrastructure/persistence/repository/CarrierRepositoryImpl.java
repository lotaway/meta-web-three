package com.metawebthree.logistics.infrastructure.persistence.repository;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.metawebthree.logistics.domain.entity.Carrier;
import com.metawebthree.logistics.infrastructure.persistence.converter.CarrierConverter;
import com.metawebthree.logistics.infrastructure.persistence.dataobject.CarrierDO;
import com.metawebthree.logistics.infrastructure.persistence.mapper.CarrierMapper;
import org.springframework.stereotype.Repository;

import java.util.List;
import java.util.Optional;
import java.util.stream.Collectors;

@Repository
public class CarrierRepositoryImpl implements CarrierRepository {

    private final CarrierMapper carrierMapper;
    private final CarrierConverter carrierConverter;

    public CarrierRepositoryImpl(CarrierMapper carrierMapper, CarrierConverter carrierConverter) {
        this.carrierMapper = carrierMapper;
        this.carrierConverter = carrierConverter;
    }

    @Override
    public Optional<Carrier> findById(Long id) {
        CarrierDO carrierDO = carrierMapper.selectById(id);
        return Optional.ofNullable(carrierConverter.toEntity(carrierDO));
    }

    @Override
    public Optional<Carrier> findByCarrierCode(String carrierCode) {
        LambdaQueryWrapper<CarrierDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(CarrierDO::getCarrierCode, carrierCode);
        CarrierDO carrierDO = carrierMapper.selectOne(wrapper);
        return Optional.ofNullable(carrierConverter.toEntity(carrierDO));
    }

    @Override
    public List<Carrier> findAll() {
        List<CarrierDO> carrierDOs = carrierMapper.selectList(null);
        return carrierDOs.stream()
            .map(carrierConverter::toEntity)
            .collect(Collectors.toList());
    }

    @Override
    public Carrier save(Carrier carrier) {
        CarrierDO carrierDO = carrierConverter.toDO(carrier);
        if (carrier.getId() == null) {
            carrierMapper.insert(carrierDO);
            carrier.setId(carrierDO.getId());
        } else {
            carrierMapper.updateById(carrierDO);
        }
        return carrier;
    }

    @Override
    public void delete(Carrier carrier) {
        if (carrier.getId() != null) {
            carrierMapper.deleteById(carrier.getId());
        }
    }
}