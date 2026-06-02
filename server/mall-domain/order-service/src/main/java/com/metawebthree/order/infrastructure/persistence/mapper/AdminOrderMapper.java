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
    
    /**
     * Get hot products based on sales count
     * @param limit max number of products to return
     * @return list of maps with productId, productName, salesCount, salesAmount
     */
    @Select("SELECT p.id as productId, p.name as productName, COUNT(oi.id) as salesCount, SUM(oi.price * oi.quantity) as salesAmount " +
            "FROM tb_product p " +
            "LEFT JOIN tb_order_item oi ON p.id = oi.product_id " +
            "LEFT JOIN tb_order o ON oi.order_id = o.id AND o.delete_status = 0 AND o.status = 3 " +
            "WHERE p.delete_status = 0 " +
            "GROUP BY p.id, p.name " +
            "ORDER BY salesCount DESC, salesAmount DESC LIMIT #{limit}")
    List<Map<String, Object>> selectHotProducts(@Param("limit") int limit);
    
    /**
     * Get sales by hour for today
     * @return list of maps with hour, sales (total amount), orders (count)
     */
    @Select("SELECT HOUR(created_at) as hour, SUM(order_amount) as sales, COUNT(*) as orders " +
            "FROM tb_order " +
            "WHERE delete_status = 0 AND status = 3 AND DATE(created_at) = CURDATE() " +
            "GROUP BY HOUR(created_at) ORDER BY hour")
    List<Map<String, Object>> selectSalesByHourToday();
}
