package com.metawebthree.mes.domain.service;

import com.metawebthree.mes.domain.entity.*;
import com.metawebthree.mes.domain.repository.*;
import org.junit.jupiter.api.Test;
import java.util.*;
import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.Mockito.*;

class WorkOrderSplitTest {
    WorkOrder parent() {
        WorkOrder p=new WorkOrder(); p.createWithType("WO-100","P","Product",100,"W","R","SPECIAL"); p.setId(1L); return p;
    }
    WorkOrderSplitRule rule(String type) {
        WorkOrderSplitRule r=new WorkOrderSplitRule(); r.create("Split","S",type); r.setId(2L); r.setSplitQuantity(30); return r;
    }
    @Test void batchUsesStoredStateAndConservesQuantity() {
        WorkOrderRepository orders=mock(WorkOrderRepository.class);
        WorkOrderSplitRuleRepository rules=mock(WorkOrderSplitRuleRepository.class);
        WorkOrder saved=parent();
        when(orders.findByIdForUpdate(1L)).thenReturn(Optional.of(saved));
        when(rules.findById(2L)).thenReturn(Optional.of(rule("BY_QUANTITY")));
        when(orders.findByParentWorkOrderId(1L)).thenReturn(List.of());
        WorkOrder untrusted=parent(); untrusted.setQuantity(1);
        List<WorkOrder> children=new WorkOrderSplitServiceImpl(rules,orders).splitWorkOrder(untrusted,2L);
        assertEquals(List.of(30,30,30,10),children.stream().map(WorkOrder::getQuantity).toList());
        assertEquals(100,saved.getQuantity()); assertEquals(WorkOrder.WorkOrderStatus.CANCELLED,saved.getStatus());
        assertTrue(children.stream().allMatch(c -> "SPECIAL".equals(c.getTypeCode()) && c.getParentWorkOrderId()==1L));
        verify(orders,times(5)).save(any());
    }
    @Test void invalidQuantitiesAndOverflowReject() {
        for(List<Integer> values:List.of(List.of(0,100),List.of(-1,101),List.of(40,50),List.of(Integer.MAX_VALUE,Integer.MAX_VALUE),List.of(100)))
            assertThrows(IllegalArgumentException.class,() -> WorkOrderPartition.preview(parent(),rule("MANUAL"),values));
        assertThrows(IllegalArgumentException.class,() -> WorkOrderPartition.preview(parent(),rule("MANUAL"),Arrays.asList(null,100)));
        assertThrows(IllegalArgumentException.class,() -> WorkOrderPartition.preview(parent(),rule("MANUAL"),null));
    }
    @Test void manualPreservesMetadata() {
        List<WorkOrder> children=WorkOrderPartition.preview(parent(),rule("MANUAL"),List.of(25,75));
        assertEquals("WO-100-S01",children.get(0).getWorkOrderNo()); assertEquals("R",children.get(0).getProcessRouteId());
        assertEquals(100,children.stream().mapToInt(WorkOrder::getQuantity).sum());
    }
    @Test void unsupportedAndMergeReject() {
        WorkOrderSplitServiceImpl s=new WorkOrderSplitServiceImpl(mock(WorkOrderSplitRuleRepository.class),mock(WorkOrderRepository.class));
        assertThrows(UnsupportedOperationException.class,() -> s.splitByBom(parent(),rule("BY_BOM")));
        assertThrows(UnsupportedOperationException.class,() -> s.splitByProcess(parent(),rule("BY_PROCESS")));
        assertThrows(UnsupportedOperationException.class,() -> s.mergeChildWorkOrders(1L));
    }
    @Test void disabledNonDraftAndCountReject() {
        WorkOrderSplitRule r=rule("MANUAL"); r.disable();
        assertThrows(IllegalArgumentException.class,() -> WorkOrderPartition.preview(parent(),r,List.of(50,50)));
        WorkOrder p=parent(); p.setStatus(WorkOrder.WorkOrderStatus.RELEASED);
        assertThrows(IllegalStateException.class,() -> WorkOrderPartition.preview(p,rule("MANUAL"),List.of(50,50)));
        WorkOrderSplitRule limited=rule("MANUAL"); limited.setMaxChildOrders(2);
        assertThrows(IllegalArgumentException.class,() -> WorkOrderPartition.preview(parent(),limited,List.of(30,30,40)));
    }
    @Test void numericConditionsAndUnsupportedLogic() {
        WorkOrderSplitRule r=rule("MANUAL"); WorkOrderSplitRule.SplitCondition c=new WorkOrderSplitRule.SplitCondition();
        c.setField("quantity"); c.setOperator("EQ"); c.setValue(100L); r.setConditions(List.of(c));
        assertTrue(WorkOrderPartition.conditionsMatch(parent(),r)); c.setOperator("GTE"); c.setValue(null);
        assertFalse(WorkOrderPartition.conditionsMatch(parent(),r)); c.setLogic("OR");
        assertThrows(IllegalArgumentException.class,() -> WorkOrderPartition.conditionsMatch(parent(),r));
    }
    @Test void previousChildrenRejectWithoutWrites() {
        WorkOrderRepository orders=mock(WorkOrderRepository.class); WorkOrderSplitRuleRepository rules=mock(WorkOrderSplitRuleRepository.class);
        when(orders.findByIdForUpdate(1L)).thenReturn(Optional.of(parent())); when(rules.findById(2L)).thenReturn(Optional.of(rule("BY_QUANTITY")));
        when(orders.findByParentWorkOrderId(1L)).thenReturn(List.of(parent()));
        assertThrows(IllegalStateException.class,() -> new WorkOrderSplitServiceImpl(rules,orders).splitWorkOrder(parent(),2L));
        verify(orders,never()).save(any());
    }
}
