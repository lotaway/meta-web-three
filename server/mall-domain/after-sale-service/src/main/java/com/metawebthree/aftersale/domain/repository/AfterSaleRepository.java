package com.metawebthree.aftersale.domain.repository;

import com.metawebthree.aftersale.application.dto.AfterSaleQueryDTO;
import com.metawebthree.aftersale.domain.model.AfterSaleOrderDO;
import java.util.List;

public interface AfterSaleRepository {
    AfterSaleOrderDO save(AfterSaleOrderDO afterSaleOrder);
    AfterSaleOrderDO findById(Long id);
    List<AfterSaleOrderDO> findByUserId(Long userId);
    List<AfterSaleOrderDO> findByOrderId(Long orderId);
    List<AfterSaleOrderDO> findAll();
    boolean updateStatus(Long id, Integer status);
    boolean deleteById(Long id);
    
    // Pagination and query support
    List<AfterSaleOrderDO> findByPage(AfterSaleQueryDTO queryDTO);
    Long countByPage(AfterSaleQueryDTO queryDTO);
    
    // Statistics
    Long countByStatus(Integer status);
    Long countTotal();
    Long sumRefundAmount();
    Long countByDateRange(String startDate, String endDate);
}