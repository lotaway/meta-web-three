package com.metawebthree.mes.application.command;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.metawebthree.mes.infrastructure.persistence.dataobject.ProcessNodeTypeDO;
import com.metawebthree.mes.infrastructure.persistence.mapper.ProcessNodeTypeMapper;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import java.time.LocalDateTime;
import java.util.List;

@Service
@RequiredArgsConstructor
public class ProcessNodeTypeService {
    
    private final ProcessNodeTypeMapper nodeTypeMapper;
    
    @Transactional
    public ProcessNodeTypeDO createNodeType(ProcessNodeTypeDO nodeType) {
        nodeType.setDeleted(false);
        nodeType.setEnabled(true);
        nodeType.setCreatedAt(LocalDateTime.now());
        nodeTypeMapper.insert(nodeType);
        return nodeType;
    }
    
    @Transactional
    public ProcessNodeTypeDO updateNodeType(ProcessNodeTypeDO nodeType) {
        nodeTypeMapper.updateById(nodeType);
        return nodeType;
    }
    
    @Transactional
    public void deleteNodeType(Long nodeTypeId) {
        ProcessNodeTypeDO nodeType = nodeTypeMapper.selectById(nodeTypeId);
        if (nodeType != null) {
            nodeType.setDeleted(true);
            nodeTypeMapper.updateById(nodeType);
        }
    }
    
    public List<ProcessNodeTypeDO> listNodeTypes(String category) {
        LambdaQueryWrapper<ProcessNodeTypeDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(ProcessNodeTypeDO::getDeleted, false);
        wrapper.eq(ProcessNodeTypeDO::getEnabled, true);
        if (category != null && !category.isEmpty()) {
            wrapper.eq(ProcessNodeTypeDO::getCategory, category);
        }
        wrapper.orderByAsc(ProcessNodeTypeDO::getSortOrder);
        return nodeTypeMapper.selectList(wrapper);
    }
    
    public ProcessNodeTypeDO getNodeType(Long nodeTypeId) {
        return nodeTypeMapper.selectById(nodeTypeId);
    }
    
    /**
     * 初始化系统预置的节点类型
     */
    @Transactional
    public void initDefaultNodeTypes() {
        if (nodeTypeMapper.selectCount(new LambdaQueryWrapper<>()) > 0) {
            return; // 已初始化
        }
        
        List<ProcessNodeTypeDO> defaultTypes = List.of(
            createNodeTypeDO("start", "开始", "START", "▶", 1),
            createNodeTypeDO("end", "结束", "END", "■", 2),
            createNodeTypeDO("manual_task", "人工任务", "TASK", "👤", 3),
            createNodeTypeDO("equipment_task", "设备交互", "EQUIPMENT", "⚙", 4),
            createNodeTypeDO("data_process", "数据处理", "SERVICE", "⚡", 5),
            createNodeTypeDO("system_integration", "系统集成", "SERVICE", "🔗", 6),
            createNodeTypeDO("exclusive_gateway", "排他网关", "GATEWAY", "◇", 7),
            createNodeTypeDO("parallel_gateway", "并行网关", "GATEWAY", "◆", 8),
            createNodeTypeDO("inclusive_gateway", "包容网关", "GATEWAY", "◈", 9),
            createNodeTypeDO("sub_process", "子流程", "SUB_PROCESS", "🔄", 10)
        );
        
        for (ProcessNodeTypeDO type : defaultTypes) {
            nodeTypeMapper.insert(type);
        }
    }
    
    private ProcessNodeTypeDO createNodeTypeDO(String code, String name, String category, String icon, int order) {
        ProcessNodeTypeDO nodeType = new ProcessNodeTypeDO();
        nodeType.setNodeTypeCode(code);
        nodeType.setNodeTypeName(name);
        nodeType.setCategory(category);
        nodeType.setIcon(icon);
        nodeType.setSortOrder(order);
        nodeType.setDescription("系统预置节点类型: " + name);
        return nodeType;
    }
}