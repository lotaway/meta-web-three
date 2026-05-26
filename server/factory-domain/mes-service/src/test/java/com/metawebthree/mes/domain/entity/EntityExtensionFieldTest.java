package com.metawebthree.mes.domain.entity;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class EntityExtensionFieldTest {
    
    @Test
    void createShouldInitializeFieldWithDefaultValues() {
        EntityExtensionField field = EntityExtensionField.create(
                "WORK_ORDER", "wo_title", "工单标题",
                EntityExtensionField.FieldType.TEXT, "basic"
        );
        
        assertEquals("WORK_ORDER", field.getEntityType());
        assertEquals("wo_title", field.getFieldCode());
        assertEquals("工单标题", field.getFieldName());
        assertEquals(EntityExtensionField.FieldType.TEXT, field.getFieldType());
        assertEquals("basic", field.getFieldGroup());
        assertFalse(field.getRequired());
        assertFalse(field.getIsUnique());
        assertTrue(field.getListVisible());
        assertFalse(field.getSearchable());
        assertEquals(EntityExtensionField.FieldStatus.ACTIVE, field.getStatus());
        assertNotNull(field.getCreatedAt());
        assertNotNull(field.getUpdatedAt());
    }
    
    @Test
    void validateValueShouldReturnTrueForValidValue() {
        EntityExtensionField field = EntityExtensionField.create(
                "WORK_ORDER", "wo_code", "工单编码",
                EntityExtensionField.FieldType.TEXT, "basic"
        );
        field.setRequired(false);
        
        assertTrue(field.validateValue("WO-001"));
    }
    
    @Test
    void validateValueShouldReturnFalseForRequiredEmptyValue() {
        EntityExtensionField field = EntityExtensionField.create(
                "WORK_ORDER", "wo_code", "工单编码",
                EntityExtensionField.FieldType.TEXT, "basic"
        );
        field.setRequired(true);
        
        assertFalse(field.validateValue(null));
        assertFalse(field.validateValue(""));
    }
    
    @Test
    void validateValueShouldValidateAgainstRegexRule() {
        EntityExtensionField field = EntityExtensionField.create(
                "WORK_ORDER", "wo_code", "工单编码",
                EntityExtensionField.FieldType.TEXT, "basic"
        );
        field.setValidationRule("^WO-\\d{3}$");
        
        assertTrue(field.validateValue("WO-001"));
        assertFalse(field.validateValue("INVALID"));
    }
}