package com.metawebthree.production.domain.repository;

import com.metawebthree.production.domain.entity.WorkStationBinding;
import java.util.List;

public interface WorkStationBindingRepository {
    void save(WorkStationBinding binding);
    void update(WorkStationBinding binding);
    void deleteById(Long id);
    WorkStationBinding findById(Long id);
    List<WorkStationBinding> findByWorkstationCode(String workstationCode);
    List<WorkStationBinding> findByWorkstationCodeAndType(String workstationCode, WorkStationBinding.BindingType type);
    List<WorkStationBinding> findByTargetCode(String targetCode);
    WorkStationBinding findPrimaryByWorkstationAndType(String workstationCode, WorkStationBinding.BindingType type);
}