package com.metawebthree.mes.infrastructure.persistence.repository.labor;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.metawebthree.mes.domain.entity.labor.Operator;
import com.metawebthree.mes.domain.entity.labor.Operator.OperatorStatus;
import com.metawebthree.mes.domain.entity.labor.OperatorSkill;
import com.metawebthree.mes.domain.entity.labor.OperatorSkill.SkillLevel;
import com.metawebthree.mes.domain.repository.labor.OperatorRepository;
import com.metawebthree.mes.infrastructure.persistence.dataobject.labor.OperatorDO;
import com.metawebthree.mes.infrastructure.persistence.dataobject.labor.OperatorSkillDO;
import com.metawebthree.mes.infrastructure.persistence.mapper.labor.OperatorMapper;
import com.metawebthree.mes.infrastructure.persistence.mapper.labor.OperatorSkillMapper;
import org.springframework.stereotype.Repository;

import java.time.LocalDateTime;
import java.util.List;
import java.util.Optional;
import java.util.stream.Collectors;

@Repository
public class OperatorRepositoryImpl implements OperatorRepository {

    private final OperatorMapper operatorMapper;
    private final OperatorSkillMapper skillMapper;

    public OperatorRepositoryImpl(OperatorMapper operatorMapper, OperatorSkillMapper skillMapper) {
        this.operatorMapper = operatorMapper;
        this.skillMapper = skillMapper;
    }

    @Override
    public Optional<Operator> findById(Long id) {
        OperatorDO doObj = operatorMapper.selectById(id);
        return Optional.ofNullable(doObj).map(this::toEntityWithSkills);
    }

    @Override
    public Optional<Operator> findByOperatorCode(String operatorCode) {
        LambdaQueryWrapper<OperatorDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(OperatorDO::getOperatorCode, operatorCode);
        OperatorDO doObj = operatorMapper.selectOne(wrapper);
        return Optional.ofNullable(doObj).map(this::toEntityWithSkills);
    }

    @Override
    public List<Operator> findByDepartment(String department) {
        LambdaQueryWrapper<OperatorDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(OperatorDO::getDepartment, department);
        List<OperatorDO> doList = operatorMapper.selectList(wrapper);
        return doList.stream().map(this::toEntityWithSkills).collect(Collectors.toList());
    }

    @Override
    public List<Operator> findByStatus(Operator.OperatorStatus status) {
        LambdaQueryWrapper<OperatorDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(OperatorDO::getStatus, status.name());
        List<OperatorDO> doList = operatorMapper.selectList(wrapper);
        return doList.stream().map(this::toEntityWithSkills).collect(Collectors.toList());
    }

    @Override
    public List<Operator> findByShiftGroup(String shiftGroup) {
        LambdaQueryWrapper<OperatorDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(OperatorDO::getShiftGroup, shiftGroup);
        List<OperatorDO> doList = operatorMapper.selectList(wrapper);
        return doList.stream().map(this::toEntityWithSkills).collect(Collectors.toList());
    }

    @Override
    public List<Operator> findAll() {
        List<OperatorDO> doList = operatorMapper.selectList(null);
        return doList.stream().map(this::toEntityWithSkills).collect(Collectors.toList());
    }

    @Override
    public Operator save(Operator entity) {
        OperatorDO doObj = toDO(entity);
        if (doObj.getId() == null) {
            operatorMapper.insert(doObj);
            entity.setId(doObj.getId());
        } else {
            operatorMapper.updateById(doObj);
        }
        saveSkills(entity.getId(), entity.getSkills());
        return entity;
    }

    @Override
    public void update(Operator entity) {
        if (entity.getId() != null) {
            OperatorDO doObj = toDO(entity);
            operatorMapper.updateById(doObj);
            saveSkills(entity.getId(), entity.getSkills());
        }
    }

    @Override
    public void deleteById(Long id) {
        LambdaQueryWrapper<OperatorSkillDO> skillWrapper = new LambdaQueryWrapper<>();
        skillWrapper.eq(OperatorSkillDO::getOperatorId, id);
        skillMapper.delete(skillWrapper);
        operatorMapper.deleteById(id);
    }

    private void saveSkills(Long operatorId, List<OperatorSkill> skills) {
        LambdaQueryWrapper<OperatorSkillDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(OperatorSkillDO::getOperatorId, operatorId);
        skillMapper.delete(wrapper);
        if (skills != null && !skills.isEmpty()) {
            for (OperatorSkill skill : skills) {
                skillMapper.insert(toSkillDO(operatorId, skill));
            }
        }
    }

    private Operator toEntityWithSkills(OperatorDO doObj) {
        Operator entity = toEntity(doObj);
        LambdaQueryWrapper<OperatorSkillDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(OperatorSkillDO::getOperatorId, doObj.getId());
        List<OperatorSkillDO> skillDOs = skillMapper.selectList(wrapper);
        List<OperatorSkill> skills = skillDOs.stream().map(this::toSkill).collect(Collectors.toList());
        entity.setSkills(skills);
        return entity;
    }

    private Operator toEntity(OperatorDO doObj) {
        if (doObj == null) return null;
        Operator entity = new Operator();
        entity.setId(doObj.getId());
        entity.setOperatorCode(doObj.getOperatorCode());
        entity.setOperatorName(doObj.getOperatorName());
        entity.setDepartment(doObj.getDepartment());
        entity.setJobTitle(doObj.getJobTitle());
        entity.setShiftGroup(doObj.getShiftGroup());
        entity.setStatus(doObj.getStatus() != null ? OperatorStatus.valueOf(doObj.getStatus()) : OperatorStatus.ACTIVE);
        entity.setPhone(doObj.getPhone());
        entity.setEmail(doObj.getEmail());
        entity.setIdCardNo(doObj.getIdCardNo());
        entity.setHireDate(doObj.getHireDate());
        entity.setRemark(doObj.getRemark());
        entity.setCreatedBy(doObj.getCreatedBy());
        entity.setCreatedAt(doObj.getCreatedAt());
        entity.setUpdatedAt(doObj.getUpdatedAt());
        return entity;
    }

    private OperatorDO toDO(Operator entity) {
        if (entity == null) return null;
        OperatorDO doObj = new OperatorDO();
        doObj.setId(entity.getId());
        doObj.setOperatorCode(entity.getOperatorCode());
        doObj.setOperatorName(entity.getOperatorName());
        doObj.setDepartment(entity.getDepartment());
        doObj.setJobTitle(entity.getJobTitle());
        doObj.setShiftGroup(entity.getShiftGroup());
        doObj.setStatus(entity.getStatus() != null ? entity.getStatus().name() : OperatorStatus.ACTIVE.name());
        doObj.setPhone(entity.getPhone());
        doObj.setEmail(entity.getEmail());
        doObj.setIdCardNo(entity.getIdCardNo());
        doObj.setHireDate(entity.getHireDate());
        doObj.setRemark(entity.getRemark());
        doObj.setCreatedBy(entity.getCreatedBy());
        doObj.setCreatedAt(entity.getCreatedAt());
        doObj.setUpdatedAt(entity.getUpdatedAt());
        return doObj;
    }

    private OperatorSkill toSkill(OperatorSkillDO doObj) {
        if (doObj == null) return null;
        OperatorSkill skill = new OperatorSkill();
        skill.setId(doObj.getId());
        skill.setOperatorId(doObj.getOperatorId());
        skill.setSkillCode(doObj.getSkillCode());
        skill.setSkillName(doObj.getSkillName());
        skill.setSkillLevel(doObj.getSkillLevel() != null ? SkillLevel.valueOf(doObj.getSkillLevel()) : null);
        skill.setCertified(doObj.getCertified() != null ? doObj.getCertified() : false);
        skill.setCertifiedAt(doObj.getCertifiedAt());
        skill.setExpiryAt(doObj.getExpiryAt());
        skill.setCreatedAt(doObj.getCreatedAt());
        skill.setUpdatedAt(doObj.getUpdatedAt());
        return skill;
    }

    private OperatorSkillDO toSkillDO(Long operatorId, OperatorSkill entity) {
        if (entity == null) return null;
        OperatorSkillDO doObj = new OperatorSkillDO();
        doObj.setId(entity.getId());
        doObj.setOperatorId(operatorId);
        doObj.setSkillCode(entity.getSkillCode());
        doObj.setSkillName(entity.getSkillName());
        doObj.setSkillLevel(entity.getSkillLevel() != null ? entity.getSkillLevel().name() : null);
        doObj.setCertified(entity.isCertified());
        doObj.setCertifiedAt(entity.getCertifiedAt());
        doObj.setExpiryAt(entity.getExpiryAt());
        doObj.setCreatedAt(entity.getCreatedAt());
        doObj.setUpdatedAt(entity.getUpdatedAt());
        return doObj;
    }
}
