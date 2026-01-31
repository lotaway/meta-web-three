package com.metawebthree.payment.infrastructure.persistence.mapper;
import com.metawebthree.payment.domain.model.*;
 
import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.metawebthree.payment.domain.model.ExchangeOrder;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;
import org.apache.ibatis.annotations.Select;

import java.math.BigDecimal;
import java.sql.Timestamp;
import java.util.List;

@Mapper
public interface ExchangeOrderRepository extends BaseMapper<ExchangeOrder> {

        @Select("SELECT * FROM Exchange_Orders WHERE order_no = #{orderNo}")
        ExchangeOrder findByOrderNo(@Param("orderNo") String orderNo);

        @Select("SELECT * FROM Exchange_Orders WHERE user_id = #{userId}")
        List<ExchangeOrder> findByUserId(@Param("userId") Long userId);

        @Select("SELECT * FROM Exchange_Orders WHERE user_id = #{userId} AND status = #{status}")
        List<ExchangeOrder> findByUserIdAndStatus(@Param("userId") Long userId, @Param("status") String status);

        @Select("SELECT * FROM Exchange_Orders WHERE status = #{status}")
        List<ExchangeOrder> findByStatus(@Param("status") String status);

        @Select("SELECT * FROM Exchange_Orders WHERE status = #{status} AND created_at < #{time}")
        List<ExchangeOrder> findByStatusAndCreatedAtBefore(@Param("status") String status,
                        @Param("time") Timestamp time);

        @Select("SELECT * FROM Exchange_Orders WHERE user_id = #{userId} AND status IN ('PENDING', 'PAID', 'PROCESSING')")
        List<ExchangeOrder> findActiveOrdersByUserId(@Param("userId") Long userId);

        @Select("SELECT COALESCE(SUM(fiat_amount), 0) FROM Exchange_Orders WHERE user_id = #{userId} AND status = 'COMPLETED' AND created_at >= #{startDate}")
        BigDecimal getTotalCompletedAmountByUserIdAndDateRange(@Param("userId") Long userId,
                        @Param("startDate") Timestamp startDate);

        @Select("SELECT COUNT(*) FROM Exchange_Orders WHERE user_id = #{userId} AND status = 'COMPLETED' AND created_at >= #{startDate}")
        Long getCompletedOrderCountByUserIdAndDateRange(@Param("userId") Long userId,
                        @Param("startDate") Timestamp startDate);

        @Select("SELECT * FROM Exchange_Orders WHERE payment_order_no = #{paymentOrderNo}")
        ExchangeOrder findByPaymentOrderNo(@Param("paymentOrderNo") String paymentOrderNo);

        @Select("SELECT * FROM Exchange_Orders WHERE crypto_transaction_hash = #{txHash}")
        ExchangeOrder findByCryptoTransactionHash(@Param("txHash") String txHash);

        @Select("SELECT COUNT(*) FROM Exchange_Orders WHERE order_no = #{orderNo}")
        boolean existsByOrderNo(@Param("orderNo") String orderNo);

        @Select("SELECT * FROM Exchange_Orders WHERE created_at BETWEEN #{startDate} AND #{endDate}")
        List<ExchangeOrder> findByCreatedAtBetween(@Param("startDate") Timestamp start,
                        @Param("endDate") Timestamp end);

        @Select("SELECT * FROM Exchange_Orders WHERE status = #{status} AND created_at BETWEEN #{startDate} AND #{endDate}")
        List<ExchangeOrder> findByStatusAndCreatedAtBetween(@Param("status") String status,
                        @Param("startDate") Timestamp start, @Param("endDate") Timestamp end);

}
