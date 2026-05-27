package com.metawebthree.production.application;

import com.metawebthree.production.domain.entity.WorkStationBinding;
import com.metawebthree.production.domain.repository.WorkStationBindingRepository;
import org.springframework.stereotype.Service;
import java.util.List;

@Service
public class WorkStationBindingService {
    
    private final WorkStationBindingRepository repository;
    
    public WorkStationBindingService(WorkStationBindingRepository repository) {
        this.repository = repository;
    }
    
    public WorkStationBinding createBinding(String workstationCode, 
                                            WorkStationBinding.BindingType bindingType,
                                            String targetCode, String targetName, String targetType) {
        WorkStationBinding binding = new WorkStationBinding();
        binding.create(workstationCode, bindingType, targetCode, targetName, targetType);
        repository.save(binding);
        return binding;
    }
    
    public WorkStationBinding bindEquipment(String workstationCode, String equipmentCode, 
                             String equipmentName, String equipmentType) {
        return createBinding(workstationCode, WorkStationBinding.BindingType.EQUIPMENT, 
                     equipmentCode, equipmentName, equipmentType);
    }
    
    public WorkStationBinding bindTool(String workstationCode, String toolCode, 
                        String toolName, String toolType) {
        return createBinding(workstationCode, WorkStationBinding.BindingType.TOOL, 
                     toolCode, toolName, toolType);
    }
    
    public WorkStationBinding bindPersonnel(String workstationCode, String personnelCode, 
                             String personnelName, String personnelType) {
        return createBinding(workstationCode, WorkStationBinding.BindingType.PERSONNEL, 
                     personnelCode, personnelName, personnelType);
    }
    
    public void setPrimaryBinding(Long bindingId) {
        WorkStationBinding binding = repository.findById(bindingId);
        if (binding == null) {
            throw new IllegalArgumentException("Binding not found");
        }
        binding.setPrimary(true);
        repository.update(binding);
    }
    
    public void unbind(Long bindingId) {
        WorkStationBinding binding = repository.findById(bindingId);
        if (binding == null) {
            throw new IllegalArgumentException("Binding not found");
        }
        binding.unbind();
        repository.update(binding);
    }
    
    public List<WorkStationBinding> getBindingsByWorkstation(String workstationCode) {
        return repository.findByWorkstationCode(workstationCode);
    }
    
    public List<WorkStationBinding> getEquipmentBindings(String workstationCode) {
        return repository.findByWorkstationCodeAndType(workstationCode, 
            WorkStationBinding.BindingType.EQUIPMENT);
    }
    
    public List<WorkStationBinding> getToolBindings(String workstationCode) {
        return repository.findByWorkstationCodeAndType(workstationCode, 
            WorkStationBinding.BindingType.TOOL);
    }
    
    public List<WorkStationBinding> getPersonnelBindings(String workstationCode) {
        return repository.findByWorkstationCodeAndType(workstationCode, 
            WorkStationBinding.BindingType.PERSONNEL);
    }
    
    public WorkStationBinding getPrimaryEquipment(String workstationCode) {
        return repository.findPrimaryByWorkstationAndType(workstationCode, 
            WorkStationBinding.BindingType.EQUIPMENT);
    }
    
    public void deleteBinding(Long bindingId) {
        repository.deleteById(bindingId);
    }
}