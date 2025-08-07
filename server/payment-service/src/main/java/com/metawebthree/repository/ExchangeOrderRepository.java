package com.metawebthree.repository;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.baomidou.mybatisplus.core.metadata.IPage;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.metawebthree.entity.ExchangeOrder;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;
import org.apache.ibatis.annotations.Select;

import java.math.BigDecimal;
import java.time.LocalDateTime;
import java.util.List;

@Mapper
public interface ExchangeOrderRepository extends BaseMapper<ExchangeOrder> {
    
    @Select("SELECT * FROM exchange_orders WHERE order_no = #{orderNo}")
    ExchangeOrder findByOrderNo(@Param("orderNo") String orderNo);
    
    @Select("SELECT * FROM exchange_orders WHERE user_id = #{userId}")
    List<ExchangeOrder> findByUserId(@Param("userId") Long userId);
    
    @Select("SELECT * FROM exchange_orders WHERE user_id = #{userId} AND status = #{status}")
    List<ExchangeOrder> findByUserIdAndStatus(@Param("userId") Long userId, @Param("status") String status);
    
    @Select("SELECT * FROM exchange_orders WHERE status = #{status}")
    List<ExchangeOrder> findByStatus(@Param("status") String status);
    
    @Select("SELECT * FROM exchange_orders WHERE status = #{status} AND created_at < #{time}")
    List<ExchangeOrder> findByStatusAndCreatedAtBefore(@Param("status") String status, @Param("time") LocalDateTime time);
    
    @Select("SELECT * FROM exchange_orders WHERE user_id = #{userId} AND status IN ('PENDING', 'PAID', 'PROCESSING')")
    List<ExchangeOrder> findActiveOrdersByUserId(@Param("userId") Long userId);
    
    @Select("SELECT COALESCE(SUM(fiat_amount), 0) FROM exchange_orders WHERE user_id = #{userId} AND status = 'COMPLETED' AND created_at >= #{startDate}")
    BigDecimal getTotalCompletedAmountByUserIdAndDateRange(@Param("userId") Long userId, @Param("startDate") LocalDateTime startDate);
    
    @Select("SELECT COUNT(*) FROM exchange_orders WHERE user_id = #{userId} AND status = 'COMPLETED' AND created_at >= #{startDate}")
    Long getCompletedOrderCountByUserIdAndDateRange(@Param("userId") Long userId, @Param("startDate") LocalDateTime startDate);
    
    @Select("SELECT * FROM exchange_orders WHERE payment_order_no = #{paymentOrderNo}")
    ExchangeOrder findByPaymentOrderNo(@Param("paymentOrderNo") String paymentOrderNo);
    
    @Select("SELECT * FROM exchange_orders WHERE crypto_transaction_hash = #{txHash}")
    ExchangeOrder findByCryptoTransactionHash(@Param("txHash") String txHash);
    
    @Select("SELECT COUNT(*) FROM exchange_orders WHERE order_no = #{orderNo}")
    boolean existsByOrderNo(@Param("orderNo") String orderNo);
} 