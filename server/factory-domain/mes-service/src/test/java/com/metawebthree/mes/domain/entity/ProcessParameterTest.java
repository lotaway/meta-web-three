package com.metawebthree.mes.domain.entity;

import org.junit.jupiter.api.Test;

import java.math.BigDecimal;

import static org.junit.jupiter.api.Assertions.*;

public class ProcessParameterTest {
    
    @Test
    void testCreateStaticFactoryMethod() {
        ProcessParameter param = ProcessParameter.create(
                "TEMP_001",
                "温度参数",
                1L,
                "ROUTE_001",
                1,
                "STEP_001",
                ProcessParameter.ParamType.TEMPERATURE,
                ProcessParameter.DataType.DECIMAL
        );
        
        assertEquals("TEMP_001", param.getParamCode());
        assertEquals("温度参数", param.getParamName());
        assertEquals(1L, param.getRouteId());
        assertEquals("ROUTE_001", param.getRouteCode());
        assertEquals(1, param.getStepNo());
        assertEquals("STEP_001", param.getStepCode());
        assertEquals(ProcessParameter.ParamType.TEMPERATURE, param.getParamType());
        assertEquals(ProcessParameter.DataType.DECIMAL, param.getDataType());
        assertEquals(false, param.getIsRequired());
        assertEquals(ProcessParameter.ParamStatus.ACTIVE, param.getStatus());
        assertNotNull(param.getCreatedAt());
        assertNotNull(param.getUpdatedAt());
    }
    
    @Test
    void testValidateValueWithinRange() {
        ProcessParameter param = ProcessParameter.create(
                "TEMP_001", "温度", 1L, "ROUTE_001", 1, "STEP_001",
                ProcessParameter.ParamType.TEMPERATURE, ProcessParameter.DataType.DECIMAL
        );
        param.setStandardValue(new BigDecimal("100"));
        param.setUpperLimit(new BigDecimal("110"));
        param.setLowerLimit(new BigDecimal("90"));
        
        assertTrue(param.validateValue(new BigDecimal("100"))); // 标准值
        assertTrue(param.validateValue(new BigDecimal("105"))); // 在范围内
        assertTrue(param.validateValue(new BigDecimal("90")));  // 下限
        assertTrue(param.validateValue(new BigDecimal("110"))); // 上限
    }
    
    @Test
    void testValidateValueOutOfRange() {
        ProcessParameter param = ProcessParameter.create(
                "TEMP_001", "温度", 1L, "ROUTE_001", 1, "STEP_001",
                ProcessParameter.ParamType.TEMPERATURE, ProcessParameter.DataType.DECIMAL
        );
        param.setStandardValue(new BigDecimal("100"));
        param.setUpperLimit(new BigDecimal("110"));
        param.setLowerLimit(new BigDecimal("90"));
        
        assertFalse(param.validateValue(new BigDecimal("115"))); // 超过上限
        assertFalse(param.validateValue(new BigDecimal("85")));  // 低于下限
    }
    
    @Test
    void testValidateValueNullValue() {
        ProcessParameter param = ProcessParameter.create(
                "TEMP_001", "温度", 1L, "ROUTE_001", 1, "STEP_001",
                ProcessParameter.ParamType.TEMPERATURE, ProcessParameter.DataType.DECIMAL
        );
        param.setIsRequired(false);
        
        // 非必填参数，null值应通过
        assertTrue(param.validateValue(null));
        
        // 必填参数，null值应不通过
        param.setIsRequired(true);
        assertFalse(param.validateValue(null));
    }
    
    @Test
    void testCalculateDeviation() {
        ProcessParameter param = ProcessParameter.create(
                "TEMP_001", "温度", 1L, "ROUTE_001", 1, "STEP_001",
                ProcessParameter.ParamType.TEMPERATURE, ProcessParameter.DataType.DECIMAL
        );
        param.setStandardValue(new BigDecimal("100"));
        
        // 偏差 10%
        BigDecimal deviation = param.calculateDeviation(new BigDecimal("110"));
        assertEquals(new BigDecimal("10.0000"), deviation);
        
        // 偏差 -5%
        deviation = param.calculateDeviation(new BigDecimal("95"));
        assertEquals(new BigDecimal("-5.0000"), deviation);
    }
    
    @Test
    void testCalculateDeviationWithNullStandardValue() {
        ProcessParameter param = ProcessParameter.create(
                "TEMP_001", "温度", 1L, "ROUTE_001", 1, "STEP_001",
                ProcessParameter.ParamType.TEMPERATURE, ProcessParameter.DataType.DECIMAL
        );
        
        assertNull(param.calculateDeviation(new BigDecimal("100")));
        assertNull(param.calculateDeviation(null));
    }
    
    @Test
    void testIsOutOfTolerance() {
        ProcessParameter param = ProcessParameter.create(
                "TEMP_001", "温度", 1L, "ROUTE_001", 1, "STEP_001",
                ProcessParameter.ParamType.TEMPERATURE, ProcessParameter.DataType.DECIMAL
        );
        param.setStandardValue(new BigDecimal("100"));
        param.setAlarmThreshold(new BigDecimal("10")); // 10%阈值
        
        assertFalse(param.isOutOfTolerance(new BigDecimal("105"))); // 5%偏差，未超差
        assertTrue(param.isOutOfTolerance(new BigDecimal("115")));  // 15%偏差，超差
        assertTrue(param.isOutOfTolerance(new BigDecimal("85")));   // -15%偏差，超差
    }
    
    @Test
    void testIsOutOfToleranceWithNullThreshold() {
        ProcessParameter param = ProcessParameter.create(
                "TEMP_001", "温度", 1L, "ROUTE_001", 1, "STEP_001",
                ProcessParameter.ParamType.TEMPERATURE, ProcessParameter.DataType.DECIMAL
        );
        param.setStandardValue(new BigDecimal("100"));
        
        // 无阈值设置，不应判定为超差
        assertFalse(param.isOutOfTolerance(new BigDecimal("200")));
    }
}