package com.metawebthree.mes.domain.entity;

import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.Optional;

import static org.junit.jupiter.api.Assertions.*;

public class ProcessRouteTest {
    
    @Test
    void testCreateProcessRoute() {
        ProcessRoute route = new ProcessRoute();
        route.create("PRoute-001", "产品A工艺路线", "P001");
        
        assertEquals("PRoute-001", route.getRouteCode());
        assertEquals("产品A工艺路线", route.getRouteName());
        assertEquals("P001", route.getProductCode());
        assertEquals(1, route.getVersion());
        assertEquals(ProcessRoute.RouteStatus.DRAFT, route.getStatus());
        assertNotNull(route.getCreatedAt());
        assertNotNull(route.getUpdatedAt());
    }
    
    @Test
    void testActivateProcessRoute() {
        ProcessRoute route = new ProcessRoute();
        route.create("PRoute-001", "产品A工艺路线", "P001");
        
        assertEquals(ProcessRoute.RouteStatus.DRAFT, route.getStatus());
        route.activate();
        assertEquals(ProcessRoute.RouteStatus.ACTIVE, route.getStatus());
    }
    
    @Test
    void testArchiveProcessRoute() {
        ProcessRoute route = new ProcessRoute();
        route.create("PRoute-001", "产品A工艺路线", "P001");
        route.activate();
        
        assertEquals(ProcessRoute.RouteStatus.ACTIVE, route.getStatus());
        route.archive();
        assertEquals(ProcessRoute.RouteStatus.ARCHIVED, route.getStatus());
    }
    
    @Test
    void testUpdateVersion() {
        ProcessRoute route = new ProcessRoute();
        route.create("PRoute-001", "产品A工艺路线", "P001");
        
        assertEquals(1, route.getVersion());
        route.updateVersion();
        assertEquals(2, route.getVersion());
        route.updateVersion();
        assertEquals(3, route.getVersion());
    }
    
    @Test
    void testProcessRouteFullLifecycle() {
        ProcessRoute route = new ProcessRoute();
        
        route.create("PRoute-001", "产品A工艺路线", "P001");
        assertEquals(ProcessRoute.RouteStatus.DRAFT, route.getStatus());
        assertEquals(1, route.getVersion());
        
        route.activate();
        assertEquals(ProcessRoute.RouteStatus.ACTIVE, route.getStatus());
        
        route.updateVersion();
        assertEquals(2, route.getVersion());
        
        route.archive();
        assertEquals(ProcessRoute.RouteStatus.ARCHIVED, route.getStatus());
    }
    
    @Test
    void testProcessRouteWithSteps() {
        ProcessRoute route = new ProcessRoute();
        route.create("PRoute-001", "产品A工艺路线", "P001");
        
        List<ProcessRoute.ProcessStep> steps = new ArrayList<>();
        
        ProcessRoute.ProcessStep step1 = new ProcessRoute.ProcessStep();
        step1.setStepNo(1);
        step1.setProcessCode("PC-001");
        step1.setProcessName("组装");
        step1.setWorkstationId(1L);
        step1.setStandardTime(300);
        step1.setQualityCheckpoint("IPQC");
        steps.add(step1);
        
        ProcessRoute.ProcessStep step2 = new ProcessRoute.ProcessStep();
        step2.setStepNo(2);
        step2.setProcessCode("PC-002");
        step2.setProcessName("测试");
        step2.setWorkstationId(2L);
        step2.setStandardTime(120);
        step2.setQualityCheckpoint("FQC");
        steps.add(step2);
        
        route.setSteps(steps);
        
        assertEquals(2, route.getSteps().size());
        assertEquals("组装", route.getSteps().get(0).getProcessName());
        assertEquals("测试", route.getSteps().get(1).getProcessName());
    }
    
    @Test
    void testMultipleActivateCalls() {
        ProcessRoute route = new ProcessRoute();
        route.create("PRoute-001", "产品A工艺路线", "P001");
        
        route.activate();
        assertEquals(ProcessRoute.RouteStatus.ACTIVE, route.getStatus());
        
        route.activate();
        assertEquals(ProcessRoute.RouteStatus.ACTIVE, route.getStatus());
    }
    
    @Test
    void testArchiveAfterActivateThenArchive() {
        ProcessRoute route = new ProcessRoute();
        route.create("PRoute-001", "产品A工艺路线", "P001");
        
        route.activate();
        route.archive();
        
        assertEquals(ProcessRoute.RouteStatus.ARCHIVED, route.getStatus());
    }
    
    @Test
    void testArchiveFromDraft() {
        ProcessRoute route = new ProcessRoute();
        route.create("PRoute-001", "产品A工艺路线", "P001");
        
        route.archive();
        assertEquals(ProcessRoute.RouteStatus.ARCHIVED, route.getStatus());
    }
    
    @Test
    void testActivateFromArchived() {
        ProcessRoute route = new ProcessRoute();
        route.create("PRoute-001", "产品A工艺路线", "P001");
        route.activate();
        route.archive();
        
        route.activate();
        assertEquals(ProcessRoute.RouteStatus.ACTIVE, route.getStatus());
    }
    
    // ========== 工序顺序校验测试 ==========
    
    @Test
    void testValidateSequence_Valid() {
        ProcessRoute route = new ProcessRoute();
        route.create("PRoute-001", "产品A工艺路线", "P001");
        
        List<ProcessRoute.ProcessStep> steps = new ArrayList<>();
        ProcessRoute.ProcessStep step1 = createStep(1, "PC-001", "组装");
        ProcessRoute.ProcessStep step2 = createStep(2, "PC-002", "测试");
        ProcessRoute.ProcessStep step3 = createStep(3, "PC-003", "包装");
        steps.add(step1);
        steps.add(step2);
        steps.add(step3);
        route.setSteps(steps);
        
        ProcessRoute.ValidationResult result = route.validateSequence();
        assertTrue(result.isValid(), "有效工序序列应该通过验证");
        assertTrue(result.getErrors().isEmpty());
    }
    
    @Test
    void testValidateSequence_DuplicateStepNo() {
        ProcessRoute route = new ProcessRoute();
        route.create("PRoute-001", "产品A工艺路线", "P001");
        
        List<ProcessRoute.ProcessStep> steps = new ArrayList<>();
        steps.add(createStep(1, "PC-001", "组装"));
        steps.add(createStep(1, "PC-002", "测试")); // 重复序号
        route.setSteps(steps);
        
        ProcessRoute.ValidationResult result = route.validateSequence();
        assertFalse(result.isValid(), "重复序号应该验证失败");
        assertTrue(result.getErrors().stream().anyMatch(e -> e.contains("重复")));
    }
    
    @Test
    void testValidateSequence_NonContinuous() {
        ProcessRoute route = new ProcessRoute();
        route.create("PRoute-001", "产品A工艺路线", "P001");
        
        List<ProcessRoute.ProcessStep> steps = new ArrayList<>();
        steps.add(createStep(1, "PC-001", "组装"));
        steps.add(createStep(3, "PC-003", "包装")); // 跳过了2
        route.setSteps(steps);
        
        ProcessRoute.ValidationResult result = route.validateSequence();
        assertFalse(result.isValid(), "不连续序号应该验证失败");
        assertTrue(result.getErrors().stream().anyMatch(e -> e.contains("不连续")));
    }
    
    @Test
    void testValidateSequence_DuplicateProcessCode() {
        ProcessRoute route = new ProcessRoute();
        route.create("PRoute-001", "产品A工艺路线", "P001");
        
        List<ProcessRoute.ProcessStep> steps = new ArrayList<>();
        ProcessRoute.ProcessStep step1 = createStep(1, "PC-001", "组装");
        ProcessRoute.ProcessStep step2 = createStep(2, "PC-001", "测试"); // 重复编码
        steps.add(step1);
        steps.add(step2);
        route.setSteps(steps);
        
        ProcessRoute.ValidationResult result = route.validateSequence();
        assertFalse(result.isValid(), "重复工序编码应该验证失败");
        assertTrue(result.getErrors().stream().anyMatch(e -> e.contains("工序编码")));
    }
    
    @Test
    void testValidateSequence_InvalidPredecessor() {
        ProcessRoute route = new ProcessRoute();
        route.create("PRoute-001", "产品A工艺路线", "P001");
        
        List<ProcessRoute.ProcessStep> steps = new ArrayList<>();
        ProcessRoute.ProcessStep step1 = createStep(1, "PC-001", "组装");
        ProcessRoute.ProcessStep step2 = createStep(2, "PC-002", "测试");
        step2.setPredecessorStepNo(5); // 不存在的前驱
        steps.add(step1);
        steps.add(step2);
        route.setSteps(steps);
        
        ProcessRoute.ValidationResult result = route.validateSequence();
        assertFalse(result.isValid(), "无效前驱应该验证失败");
        assertTrue(result.getErrors().stream().anyMatch(e -> e.contains("前驱")));
    }
    
    @Test
    void testValidateSequence_EmptySteps() {
        ProcessRoute route = new ProcessRoute();
        route.create("PRoute-001", "产品A工艺路线", "P001");
        route.setSteps(new ArrayList<>());
        
        ProcessRoute.ValidationResult result = route.validateSequence();
        assertFalse(result.isValid(), "空工序列表应该验证失败");
    }
    
    @Test
    void testGetNextStep() {
        ProcessRoute route = new ProcessRoute();
        route.create("PRoute-001", "产品A工艺路线", "P001");
        
        List<ProcessRoute.ProcessStep> steps = new ArrayList<>();
        steps.add(createStep(1, "PC-001", "组装"));
        steps.add(createStep(2, "PC-002", "测试"));
        steps.add(createStep(3, "PC-003", "包装"));
        route.setSteps(steps);
        
        Optional<ProcessRoute.ProcessStep> nextStep = route.getNextStep(1);
        assertTrue(nextStep.isPresent());
        assertEquals("测试", nextStep.get().getProcessName());
    }
    
    @Test
    void testGetNextStep_LastStep() {
        ProcessRoute route = new ProcessRoute();
        route.create("PRoute-001", "产品A工艺路线", "P001");
        
        List<ProcessRoute.ProcessStep> steps = new ArrayList<>();
        steps.add(createStep(1, "PC-001", "组装"));
        steps.add(createStep(2, "PC-002", "测试"));
        route.setSteps(steps);
        
        Optional<ProcessRoute.ProcessStep> nextStep = route.getNextStep(2);
        assertFalse(nextStep.isPresent(), "最后一步没有下一步");
    }
    
    @Test
    void testGetFirstAndLastStep() {
        ProcessRoute route = new ProcessRoute();
        route.create("PRoute-001", "产品A工艺路线", "P001");
        
        List<ProcessRoute.ProcessStep> steps = new ArrayList<>();
        steps.add(createStep(1, "PC-001", "组装"));
        steps.add(createStep(2, "PC-002", "测试"));
        steps.add(createStep(3, "PC-003", "包装"));
        route.setSteps(steps);
        
        Optional<ProcessRoute.ProcessStep> first = route.getFirstStep();
        assertTrue(first.isPresent());
        assertEquals("组装", first.get().getProcessName());
        
        Optional<ProcessRoute.ProcessStep> last = route.getLastStep();
        assertTrue(last.isPresent());
        assertEquals("包装", last.get().getProcessName());
    }
    
    @Test
    void testGetStepByNo() {
        ProcessRoute route = new ProcessRoute();
        route.create("PRoute-001", "产品A工艺路线", "P001");
        
        List<ProcessRoute.ProcessStep> steps = new ArrayList<>();
        steps.add(createStep(1, "PC-001", "组装"));
        steps.add(createStep(2, "PC-002", "测试"));
        route.setSteps(steps);
        
        Optional<ProcessRoute.ProcessStep> step = route.getStepByNo(2);
        assertTrue(step.isPresent());
        assertEquals("测试", step.get().getProcessName());
        
        Optional<ProcessRoute.ProcessStep> notFound = route.getStepByNo(99);
        assertFalse(notFound.isPresent());
    }
    
    // 辅助方法：创建工序
    private ProcessRoute.ProcessStep createStep(int stepNo, String processCode, String processName) {
        ProcessRoute.ProcessStep step = new ProcessRoute.ProcessStep();
        step.setStepNo(stepNo);
        step.setProcessCode(processCode);
        step.setProcessName(processName);
        step.setWorkstationId((long) stepNo);
        step.setStandardTime(300);
        return step;
    }
}