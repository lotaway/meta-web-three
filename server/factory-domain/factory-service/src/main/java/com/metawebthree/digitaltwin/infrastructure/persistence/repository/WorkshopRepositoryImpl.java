package com.metawebthree.digitaltwin.infrastructure.persistence.repository;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.core.metadata.IPage;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.metawebthree.digitaltwin.domain.entity.Workshop;
import com.metawebthree.digitaltwin.domain.repository.WorkshopRepository;
import com.metawebthree.digitaltwin.infrastructure.persistence.converter.WorkshopConverter;
import com.metawebthree.digitaltwin.infrastructure.persistence.dataobject.WorkshopDO;
import com.metawebthree.digitaltwin.infrastructure.persistence.mapper.WorkshopMapper;
import org.springframework.stereotype.Repository;
import java.util.List;
import java.util.Optional;
import java.util.stream.Collectors;

@Repository
public class WorkshopRepositoryImpl implements WorkshopRepository {

    private final WorkshopMapper workshopMapper;

    public WorkshopRepositoryImpl(WorkshopMapper workshopMapper) {
        this.workshopMapper = workshopMapper;
    }

    @Override
    public Optional<Workshop> findById(Long id) {
        return Optional.ofNullable(workshopMapper.selectById(id))
                .map(WorkshopConverter::toEntity);
    }

    @Override
    public Optional<Workshop> findByWorkshopCode(String workshopCode) {
        WorkshopDO d = workshopMapper.selectOne(
                new LambdaQueryWrapper<WorkshopDO>().eq(WorkshopDO::getWorkshopCode, workshopCode));
        return Optional.ofNullable(WorkshopConverter.toEntity(d));
    }

    @Override
    public List<Workshop> findByStatus(Workshop.WorkshopStatus status) {
        return workshopMapper.selectList(
                new LambdaQueryWrapper<WorkshopDO>().eq(WorkshopDO::getStatus, status.name()))
                .stream().map(WorkshopConverter::toEntity).collect(Collectors.toList());
    }

    @Override
    public List<Workshop> findAll() {
        return workshopMapper.selectList(null)
                .stream().map(WorkshopConverter::toEntity).collect(Collectors.toList());
    }

    @Override
    public IPage<Workshop> findPaginated(int page, int size) {
        Page<WorkshopDO> pageObj = new Page<>(page, size);
        IPage<WorkshopDO> result = workshopMapper.selectPage(pageObj, null);
        return result.convert(WorkshopConverter::toEntity);
    }

    @Override
    public Workshop save(Workshop workshop) {
        WorkshopDO d = WorkshopConverter.toDO(workshop);
        workshopMapper.insert(d);
        workshop.setId(d.getId());
        return workshop;
    }

    @Override
    public void update(Workshop workshop) {
        WorkshopDO d = WorkshopConverter.toDO(workshop);
        workshopMapper.updateById(d);
    }

    @Override
    public void deleteById(Long id) {
        workshopMapper.deleteById(id);
    }
}
