package com.metawebthree.mes.domain.entity;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class CodeRuleTest {
    
    @Test
    void createShouldInitializeRuleWithDefaultValues() {
        CodeRule rule = CodeRule.create("WO", "工单编码", "WORK_ORDER", "WO-[SEQ:4]", 4);
        
        assertEquals("WO", rule.getRuleCode());
        assertEquals("工单编码", rule.getRuleName());
        assertEquals("WORK_ORDER", rule.getBusinessType());
        assertEquals("WO-[SEQ:4]", rule.getRuleExpression());
        assertEquals(4, rule.getPaddingLength());
        assertEquals(1L, rule.getStartValue());
        assertEquals(1L, rule.getCurrentValue());
        assertEquals(1, rule.getStep());
        assertEquals(CodeRule.RuleStatus.ACTIVE, rule.getStatus());
        assertNotNull(rule.getCreatedAt());
    }
    
    @Test
    void addElementShouldAddElementToRule() {
        CodeRule rule = CodeRule.create("WO", "工单编码", "WORK_ORDER", "WO-[SEQ:4]", 4);
        
        CodeRule.RuleElement element = rule.addElement(
                CodeRule.RuleElement.ElementType.PREFIX, "WO-"
        );
        
        assertEquals(CodeRule.RuleElement.ElementType.PREFIX, element.getType());
        assertEquals("WO-", element.getValue());
        assertEquals(1, rule.getElements().size());
    }
    
    @Test
    void generateNextCodeShouldIncrementSequence() {
        CodeRule rule = CodeRule.create("WO", "工单编码", "WORK_ORDER", "[SEQ:4]", 4);
        rule.addElement(CodeRule.RuleElement.ElementType.SEQUENCE, null);
        
        String firstCode = rule.generateNextCode();
        String secondCode = rule.generateNextCode();
        
        assertEquals("0001", firstCode);
        assertEquals("0002", secondCode);
        assertEquals(3L, rule.getCurrentValue());
    }
    
    @Test
    void generateNextCodeShouldCombineMultipleElements() {
        CodeRule rule = CodeRule.create("WO", "工单编码", "WORK_ORDER", null, 4);
        rule.addElement(CodeRule.RuleElement.ElementType.PREFIX, "WO-");
        rule.addElement(CodeRule.RuleElement.ElementType.SEQUENCE, null);
        
        String code = rule.generateNextCode();
        
        assertTrue(code.startsWith("WO-"));
        assertEquals("WO-0001", code);
    }
    
    @Test
    void generateNextCodeShouldFormatDateElements() {
        CodeRule rule = CodeRule.create("WO", "工单编码", "WORK_ORDER", null, 4);
        rule.addElement(CodeRule.RuleElement.ElementType.DATE, "YYYYMMDD");
        rule.addElement(CodeRule.RuleElement.ElementType.SEQUENCE, null);
        
        String code = rule.generateNextCode();
        
        assertTrue(code.matches("^\\d{8}0001$"));
    }
    
    @Test
    void resetSequenceShouldResetCurrentValueToStartValue() {
        CodeRule rule = CodeRule.create("WO", "工单编码", "WORK_ORDER", "[SEQ:4]", 4);
        rule.addElement(CodeRule.RuleElement.ElementType.SEQUENCE, null);
        rule.generateNextCode();
        rule.generateNextCode();
        
        rule.resetSequence();
        
        assertEquals(1L, rule.getCurrentValue());
    }
}