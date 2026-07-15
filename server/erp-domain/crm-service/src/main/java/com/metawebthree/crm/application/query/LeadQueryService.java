package com.metawebthree.crm.application.query;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.metawebthree.crm.domain.entity.Lead;
import com.metawebthree.crm.domain.repository.LeadRepository;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;

import java.util.List;

@Service
@RequiredArgsConstructor
public class LeadQueryService {

    private final LeadRepository leadRepository;

    public Lead getById(Long id) {
        return leadRepository.selectById(id);
    }

    public List<Lead> listAll() {
        return leadRepository.selectList(null);
    }

    public List<Lead> listByStatus(String status) {
        LambdaQueryWrapper<Lead> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(Lead::getStatus, status)
               .orderByAsc(Lead::getLeadNo);
        return leadRepository.selectList(wrapper);
    }

    public List<Lead> listBySource(String source) {
        LambdaQueryWrapper<Lead> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(Lead::getSource, source);
        return leadRepository.selectList(wrapper);
    }

    public List<Lead> listByAssignedTo(String assignedTo) {
        LambdaQueryWrapper<Lead> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(Lead::getAssignedTo, assignedTo);
        return leadRepository.selectList(wrapper);
    }

    public List<Lead> search(String keywords) {
        LambdaQueryWrapper<Lead> wrapper = new LambdaQueryWrapper<>();
        wrapper.and(w -> w.like(Lead::getName, keywords)
                          .or().like(Lead::getCompany, keywords)
                          .or().like(Lead::getEmail, keywords)
                          .or().like(Lead::getPhone, keywords)
                          .or().like(Lead::getMobile, keywords))
               .orderByAsc(Lead::getLeadNo);
        return leadRepository.selectList(wrapper);
    }
}
