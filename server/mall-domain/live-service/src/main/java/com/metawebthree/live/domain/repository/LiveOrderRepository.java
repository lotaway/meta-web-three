package com.metawebthree.live.domain.repository;

import com.metawebthree.live.domain.model.LiveOrder;

import java.util.List;

public interface LiveOrderRepository {
    LiveOrder save(LiveOrder order);
    LiveOrder findById(Long id);
    LiveOrder findByOrderId(Long orderId);
    List<LiveOrder> findByRoomId(Long roomId);
    List<LiveOrder> findByUserId(Long userId);
    List<LiveOrder> findAll();
    void deleteById(Long id);
}