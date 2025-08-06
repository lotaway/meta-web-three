package com.metawebthree.repository;

import com.metawebthree.entity.ExchangeOrder;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;
import org.springframework.stereotype.Repository;

import java.math.BigDecimal;
import java.time.LocalDateTime;
import java.util.List;
import java.util.Optional;

@Repository
public interface ExchangeOrderRepository extends JpaRepository<ExchangeOrder, Long> {
    
    Optional<ExchangeOrder> findByOrderNo(String orderNo);
    
    List<ExchangeOrder> findByUserId(Long userId);
    
    List<ExchangeOrder> findByUserIdAndStatus(Long userId, ExchangeOrder.OrderStatus status);
    
    List<ExchangeOrder> findByStatus(ExchangeOrder.OrderStatus status);
    
    List<ExchangeOrder> findByStatusAndCreatedAtBefore(ExchangeOrder.OrderStatus status, LocalDateTime time);
    
    @Query("SELECT o FROM ExchangeOrder o WHERE o.userId = :userId AND o.status IN ('PENDING', 'PAID', 'PROCESSING')")
    List<ExchangeOrder> findActiveOrdersByUserId(@Param("userId") Long userId);
    
    @Query("SELECT SUM(o.fiatAmount) FROM ExchangeOrder o WHERE o.userId = :userId AND o.status = 'COMPLETED' AND o.createdAt >= :startDate")
    BigDecimal getTotalCompletedAmountByUserIdAndDateRange(@Param("userId") Long userId, @Param("startDate") LocalDateTime startDate);
    
    @Query("SELECT COUNT(o) FROM ExchangeOrder o WHERE o.userId = :userId AND o.status = 'COMPLETED' AND o.createdAt >= :startDate")
    Long getCompletedOrderCountByUserIdAndDateRange(@Param("userId") Long userId, @Param("startDate") LocalDateTime startDate);
    
    @Query("SELECT o FROM ExchangeOrder o WHERE o.paymentOrderNo = :paymentOrderNo")
    Optional<ExchangeOrder> findByPaymentOrderNo(@Param("paymentOrderNo") String paymentOrderNo);
    
    @Query("SELECT o FROM ExchangeOrder o WHERE o.cryptoTransactionHash = :txHash")
    Optional<ExchangeOrder> findByCryptoTransactionHash(@Param("txHash") String txHash);
    
    boolean existsByOrderNo(String orderNo);
} 