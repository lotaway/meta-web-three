package com.metawebthree.mes.infrastructure.persistence.repository;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.metawebthree.mes.domain.entity.SpcControlChart;
import com.metawebthree.mes.domain.entity.SpcControlChart.AlarmRule;
import com.metawebthree.mes.domain.entity.SpcControlChart.ChartType;
import com.metawebthree.mes.domain.entity.SpcControlChart.ControlLimits;
import com.metawebthree.mes.domain.entity.SpcControlChart.SamplingConfig;
import com.metawebthree.mes.domain.repository.SpcControlChartRepository;
import com.metawebthree.mes.infrastructure.persistence.dataobject.SpcControlChartDO;
import com.metawebthree.mes.infrastructure.persistence.mapper.SpcControlChartMapper;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Repository;

import java.util.List;
import java.util.Optional;
import java.util.stream.Collectors;

@Repository
public class SpcControlChartRepositoryImpl implements SpcControlChartRepository {
    
    @Autowired
    private SpcControlChartMapper mapper;
    
    private final ObjectMapper objectMapper = new ObjectMapper();
    
    @Override
    public Optional<SpcControlChart> findById(Long id) {
        SpcControlChartDO doObj = mapper.selectById(id);
        return Optional.ofNullable(doObj).map(this::toEntity);
    }
    
    @Override
    public Optional<SpcControlChart> findByChartCode(String chartCode) {
        LambdaQueryWrapper<SpcControlChartDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(SpcControlChartDO::getChartCode, chartCode);
        SpcControlChartDO doObj = mapper.selectOne(wrapper);
        return Optional.ofNullable(doObj).map(this::toEntity);
    }
    
    @Override
    public List<SpcControlChart> findAll() {
        List<SpcControlChartDO> doList = mapper.selectList(null);
        return doList.stream().map(this::toEntity).collect(Collectors.toList());
    }
    
    @Override
    public List<SpcControlChart> findByChartType(ChartType chartType) {
        LambdaQueryWrapper<SpcControlChartDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(SpcControlChartDO::getChartType, chartType.name());
        List<SpcControlChartDO> doList = mapper.selectList(wrapper);
        return doList.stream().map(this::toEntity).collect(Collectors.toList());
    }
    
    @Override
    public List<SpcControlChart> findByIsEnabled(Boolean isEnabled) {
        LambdaQueryWrapper<SpcControlChartDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(SpcControlChartDO::getIsEnabled, isEnabled);
        List<SpcControlChartDO> doList = mapper.selectList(wrapper);
        return doList.stream().map(this::toEntity).collect(Collectors.toList());
    }
    
    @Override
    public SpcControlChart save(SpcControlChart chart) {
        SpcControlChartDO doObj = toDO(chart);
        if (chart.getId() == null) {
            mapper.insert(doObj);
            chart.setId(doObj.getId());
        } else {
            mapper.updateById(doObj);
        }
        return chart;
    }
    
    @Override
    public void update(SpcControlChart chart) {
        if (chart.getId() != null) {
            SpcControlChartDO doObj = toDO(chart);
            mapper.updateById(doObj);
        }
    }
    
    @Override
    public void deleteById(Long id) {
        mapper.deleteById(id);
    }
    
    @Override
    public Boolean existsByChartCode(String chartCode) {
        LambdaQueryWrapper<SpcControlChartDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(SpcControlChartDO::getChartCode, chartCode);
        return mapper.selectCount(wrapper) > 0;
    }
    
    private SpcControlChart toEntity(SpcControlChartDO doObj) {
        if (doObj == null) {
            return null;
        }
        SpcControlChart entity = new SpcControlChart();
        entity.setId(doObj.getId());
        entity.setChartCode(doObj.getChartCode());
        entity.setChartName(doObj.getChartName());
        if (doObj.getChartType() != null) {
            entity.setChartType(ChartType.valueOf(doObj.getChartType()));
        }
        entity.setParameterCode(doObj.getParameterCode());
        entity.setIsEnabled(doObj.getIsEnabled());
        
        if (doObj.getLimitsJson() != null) {
            try {
                entity.setLimits(objectMapper.readValue(doObj.getLimitsJson(), ControlLimits.class));
            } catch (JsonProcessingException e) {
                entity.setLimits(new ControlLimits());
            }
        }
        
        if (doObj.getAlarmRulesJson() != null) {
            try {
                entity.setAlarmRules(objectMapper.readValue(doObj.getAlarmRulesJson(), 
                    new TypeReference<List<AlarmRule>>() {}));
            } catch (JsonProcessingException e) {
                entity.setAlarmRules(List.of());
            }
        }
        
        if (doObj.getSamplingConfigJson() != null) {
            try {
                entity.setSamplingConfig(objectMapper.readValue(doObj.getSamplingConfigJson(), SamplingConfig.class));
            } catch (JsonProcessingException e) {
                entity.setSamplingConfig(new SamplingConfig());
            }
        }
        
        return entity;
    }
    
    private SpcControlChartDO toDO(SpcControlChart entity) {
        if (entity == null) {
            return null;
        }
        SpcControlChartDO doObj = new SpcControlChartDO();
        doObj.setId(entity.getId());
        doObj.setChartCode(entity.getChartCode());
        doObj.setChartName(entity.getChartName());
        doObj.setChartType(entity.getChartType() != null ? entity.getChartType().name() : null);
        doObj.setParameterCode(entity.getParameterCode());
        doObj.setIsEnabled(entity.getIsEnabled());
        
        if (entity.getLimits() != null) {
            try {
                doObj.setLimitsJson(objectMapper.writeValueAsString(entity.getLimits()));
            } catch (JsonProcessingException e) {
                doObj.setLimitsJson("{}");
            }
        }
        
        if (entity.getAlarmRules() != null) {
            try {
                doObj.setAlarmRulesJson(objectMapper.writeValueAsString(entity.getAlarmRules()));
            } catch (JsonProcessingException e) {
                doObj.setAlarmRulesJson("[]");
            }
        }
        
        if (entity.getSamplingConfig() != null) {
            try {
                doObj.setSamplingConfigJson(objectMapper.writeValueAsString(entity.getSamplingConfig()));
            } catch (JsonProcessingException e) {
                doObj.setSamplingConfigJson("{}");
            }
        }
        
        doObj.setCreatedAt(entity.getCreatedAt());
        doObj.setUpdatedAt(entity.getUpdatedAt());
        return doObj;
    }
}