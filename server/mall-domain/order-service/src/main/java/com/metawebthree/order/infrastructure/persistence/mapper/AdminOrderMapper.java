package com.metawebthree.order.infrastructure.persistence.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.metawebthree.order.domain.model.AdminOrderDO;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;
import org.apache.ibatis.annotations.Select;

import java.util.List;
import java.util.Map;

@Mapper
public interface AdminOrderMapper extends BaseMapper<AdminOrderDO> {
    
    /**
     * Get order counts grouped by status
     * @return list of maps with status and count
     */
    @Select("SELECT status, COUNT(*) as count FROM tb_order WHERE delete_status = 0 GROUP BY status")
    List<Map<String, Object>> selectOrderCountGroupByStatus();
    
    /**
     * Get pending orders count (status in 0,1,2)
     * Status: 0->pending, 1->processed, 2->shipped, 3->completed, 4->closed
     * @return count of pending orders
     */
    @Select("SELECT COUNT(*) FROM tb_order WHERE delete_status = 0 AND status IN (0, 1, 2)")
    Long selectPendingOrdersCount();
    
    /**
     * Get pending payments count (status = 0, waiting for payment)
     * @return count of orders awaiting payment
     */
    @Select("SELECT COUNT(*) FROM tb_order WHERE delete_status = 0 AND status = 0")
    Long selectPendingPaymentsCount();
}
