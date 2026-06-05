package com.metawebthree.mes.infrastructure.persistence.repository.labor;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.metawebthree.mes.domain.entity.labor.Attendance;
import com.metawebthree.mes.domain.entity.labor.Attendance.AttendanceStatus;
import com.metawebthree.mes.domain.repository.labor.AttendanceRepository;
import com.metawebthree.mes.infrastructure.persistence.dataobject.labor.AttendanceDO;
import com.metawebthree.mes.infrastructure.persistence.mapper.labor.AttendanceMapper;
import org.springframework.stereotype.Repository;

import java.time.LocalDate;
import java.util.List;
import java.util.Optional;
import java.util.stream.Collectors;

@Repository
public class AttendanceRepositoryImpl implements AttendanceRepository {

    private final AttendanceMapper mapper;

    public AttendanceRepositoryImpl(AttendanceMapper mapper) {
        this.mapper = mapper;
    }

    @Override
    public Optional<Attendance> findById(Long id) {
        AttendanceDO doObj = mapper.selectById(id);
        return Optional.ofNullable(doObj).map(this::toEntity);
    }

    @Override
    public Optional<Attendance> findByOperatorIdAndDate(Long operatorId, LocalDate date) {
        LambdaQueryWrapper<AttendanceDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(AttendanceDO::getOperatorId, operatorId);
        wrapper.eq(AttendanceDO::getAttendanceDate, date);
        AttendanceDO doObj = mapper.selectOne(wrapper);
        return Optional.ofNullable(doObj).map(this::toEntity);
    }

    @Override
    public List<Attendance> findByOperatorId(Long operatorId) {
        LambdaQueryWrapper<AttendanceDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(AttendanceDO::getOperatorId, operatorId);
        return mapper.selectList(wrapper).stream().map(this::toEntity).collect(Collectors.toList());
    }

    @Override
    public List<Attendance> findByDate(LocalDate date) {
        LambdaQueryWrapper<AttendanceDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(AttendanceDO::getAttendanceDate, date);
        return mapper.selectList(wrapper).stream().map(this::toEntity).collect(Collectors.toList());
    }

    @Override
    public List<Attendance> findByDateRange(LocalDate start, LocalDate end) {
        LambdaQueryWrapper<AttendanceDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.between(AttendanceDO::getAttendanceDate, start, end);
        return mapper.selectList(wrapper).stream().map(this::toEntity).collect(Collectors.toList());
    }

    @Override
    public List<Attendance> findByStatus(AttendanceStatus status) {
        LambdaQueryWrapper<AttendanceDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(AttendanceDO::getStatus, status.name());
        return mapper.selectList(wrapper).stream().map(this::toEntity).collect(Collectors.toList());
    }

    @Override
    public List<Attendance> findAll() {
        return mapper.selectList(null).stream().map(this::toEntity).collect(Collectors.toList());
    }

    @Override
    public Attendance save(Attendance entity) {
        AttendanceDO doObj = toDO(entity);
        if (doObj.getId() == null) {
            mapper.insert(doObj);
            entity.setId(doObj.getId());
        } else {
            mapper.updateById(doObj);
        }
        return entity;
    }

    @Override
    public void update(Attendance entity) {
        if (entity.getId() != null) {
            mapper.updateById(toDO(entity));
        }
    }

    @Override
    public void deleteById(Long id) {
        mapper.deleteById(id);
    }

    private Attendance toEntity(AttendanceDO doObj) {
        if (doObj == null) return null;
        Attendance entity = new Attendance();
        entity.setId(doObj.getId());
        entity.setOperatorId(doObj.getOperatorId());
        entity.setOperatorCode(doObj.getOperatorCode());
        entity.setOperatorName(doObj.getOperatorName());
        entity.setAttendanceDate(doObj.getAttendanceDate());
        entity.setClockIn(doObj.getClockIn());
        entity.setClockOut(doObj.getClockOut());
        entity.setScheduledStart(doObj.getScheduledStart());
        entity.setScheduledEnd(doObj.getScheduledEnd());
        entity.setStatus(doObj.getStatus() != null ? AttendanceStatus.valueOf(doObj.getStatus()) : AttendanceStatus.ABSENT);
        entity.setOvertime(doObj.getOvertime() != null ? doObj.getOvertime() : false);
        entity.setRemark(doObj.getRemark());
        entity.setCreatedBy(doObj.getCreatedBy());
        entity.setCreatedAt(doObj.getCreatedAt());
        entity.setUpdatedAt(doObj.getUpdatedAt());
        return entity;
    }

    private AttendanceDO toDO(Attendance entity) {
        if (entity == null) return null;
        AttendanceDO doObj = new AttendanceDO();
        doObj.setId(entity.getId());
        doObj.setOperatorId(entity.getOperatorId());
        doObj.setOperatorCode(entity.getOperatorCode());
        doObj.setOperatorName(entity.getOperatorName());
        doObj.setAttendanceDate(entity.getAttendanceDate());
        doObj.setClockIn(entity.getClockIn());
        doObj.setClockOut(entity.getClockOut());
        doObj.setScheduledStart(entity.getScheduledStart());
        doObj.setScheduledEnd(entity.getScheduledEnd());
        doObj.setStatus(entity.getStatus() != null ? entity.getStatus().name() : AttendanceStatus.ABSENT.name());
        doObj.setOvertime(entity.isOvertime());
        doObj.setRemark(entity.getRemark());
        doObj.setCreatedBy(entity.getCreatedBy());
        doObj.setCreatedAt(entity.getCreatedAt());
        doObj.setUpdatedAt(entity.getUpdatedAt());
        return doObj;
    }
}
