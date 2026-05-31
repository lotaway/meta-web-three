package com.metawebthree.inventory.infrastructure.persistence.repository;

import com.metawebthree.inventory.domain.entity.ReservationRecord;
import com.metawebthree.inventory.infrastructure.persistence.converter.ReservationRecordConverter;
import com.metawebthree.inventory.infrastructure.persistence.dataobject.ReservationRecordDO;
import com.metawebthree.inventory.infrastructure.persistence.mapper.ReservationRecordMapper;
import org.springframework.stereotype.Repository;
import java.time.LocalDateTime;
import java.util.Optional;

@Repository
public class ReservationRepositoryImpl implements ReservationRepository {

    private final ReservationRecordMapper reservationRecordMapper;
    private final ReservationRecordConverter converter;

    public ReservationRepositoryImpl(ReservationRecordMapper reservationRecordMapper,
                                     ReservationRecordConverter converter) {
        this.reservationRecordMapper = reservationRecordMapper;
        this.converter = converter;
    }

    @Override
    public Optional<ReservationRecord> findByBizId(String bizId) {
        return reservationRecordMapper.findByBizId(bizId)
                .map(converter::toEntity);
    }

    @Override
    public ReservationRecord save(ReservationRecord record) {
        ReservationRecordDO doObj = converter.toDO(record);
        if (record.getId() == null) {
            doObj.setCreatedAt(LocalDateTime.now());
            doObj.setUpdatedAt(LocalDateTime.now());
            reservationRecordMapper.insert(doObj);
        } else {
            doObj.setUpdatedAt(LocalDateTime.now());
            reservationRecordMapper.updateById(doObj);
        }
        return converter.toEntity(doObj);
    }

    @Override
    public void delete(ReservationRecord record) {
        reservationRecordMapper.deleteById(record.getId());
    }
}