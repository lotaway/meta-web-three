package com.metawebthree.mes.domain.service;

import com.metawebthree.mes.domain.entity.WorkOrder;
import com.metawebthree.mes.domain.entity.WorkOrderSplitRule;
import java.util.List;
import java.util.Map;

public interface WorkOrderSplitService {
    /**
     * 执行工单拆分
     * @param parentWorkOrder 父工单
     * @param splitRuleId 拆分规则ID
     * @return 拆分后的子工单列表
     */
    List<WorkOrder> splitWorkOrder(WorkOrder parentWorkOrder, Long splitRuleId);
    
    /**
     * 按BOM结构拆分工单
     * @param parentWorkOrder 父工单
     * @param splitRule 拆分规则
     * @return 子工单列表
     */
    List<WorkOrder> splitByBom(WorkOrder parentWorkOrder, WorkOrderSplitRule splitRule);
    
    /**
     * 按工序拆分工单
     * @param parentWorkOrder 父工单
     * @param splitRule 拆分规则
     * @return 子工单列表
     */
    List<WorkOrder> splitByProcess(WorkOrder parentWorkOrder, WorkOrderSplitRule splitRule);
    
    /**
     * 手动拆分工单
     * @param parentWorkOrder 父工单
     * @param splitRule 拆分规则
     * @param quantities 每个子工单的数量
     * @return 子工单列表
     */
    List<WorkOrder> splitManually(WorkOrder parentWorkOrder, WorkOrderSplitRule splitRule, 
                                   List<Integer> quantities);
    
    /**
     * 获取工单的所有子工单
     * @param parentWorkOrderId 父工单ID
     * @return 子工单列表
     */
    List<WorkOrder> getChildWorkOrders(Long parentWorkOrderId);
    
    /**
     * 合并子工单回父工单
     * @param parentWorkOrderId 父工单ID
     * @return 合并后的父工单
     */
    WorkOrder mergeChildWorkOrders(Long parentWorkOrderId);
    
    /**
     * 验证拆分条件是否满足
     * @param workOrder 工单
     * @param splitRule 拆分规则
     * @return 是否满足条件
     */
    boolean validateSplitConditions(WorkOrder workOrder, WorkOrderSplitRule splitRule);
}