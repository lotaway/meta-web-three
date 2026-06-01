package com.metawebthree.aftersale.infrastructure.persistence.repository;

import com.metawebthree.aftersale.domain.model.AfterSaleOrderDO;
import com.metawebthree.aftersale.domain.repository.AfterSaleRepository;
import com.metawebthree.aftersale.infrastructure.persistence.mapper.AfterSaleMapper;
import org.springframework.stereotype.Repository;

import java.util.List;

@Repository
public class AfterSaleRepositoryImpl implements AfterSaleRepository {

    private final AfterSaleMapper afterSaleMapper;

    public AfterSaleRepositoryImpl(AfterSaleMapper afterSaleMapper) {
        this.afterSaleMapper = afterSaleMapper;
    }

    @Override
    public AfterSaleOrderDO save(AfterSaleOrderDO afterSaleOrder) {
        if (afterSaleOrder.getId() == null) {
            afterSaleMapper.insert(afterSaleOrder);
        } else {
            afterSaleMapper.update(afterSaleOrder);
        }
        return afterSaleOrder;
    }

    @Override
    public AfterSaleOrderDO findById(Long id) {
        return afterSaleMapper.selectById(id);
    }

    @Override
    public List<AfterSaleOrderDO> findByUserId(Long userId) {
        return afterSaleMapper.selectByUserId(userId);
    }

    @Override
    public List<AfterSaleOrderDO> findByOrderId(Long orderId) {
        return afterSaleMapper.selectByOrderId(orderId);
    }

    @Override
    public List<AfterSaleOrderDO> findAll() {
        return afterSaleMapper.selectAll();
    }

    @Override
    public boolean updateStatus(Long id, Integer status) {
        return afterSaleMapper.updateStatus(id, status) > 0;
    }

    @Override
    public boolean deleteById(Long id) {
        return afterSaleMapper.deleteById(id) > 0;
    }
}