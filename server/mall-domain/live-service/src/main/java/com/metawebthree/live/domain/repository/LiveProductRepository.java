package com.metawebthree.live.domain.repository;

import com.metawebthree.live.domain.model.LiveProduct;

import java.util.List;

public interface LiveProductRepository {
    LiveProduct save(LiveProduct product);
    LiveProduct findById(Long id);
    List<LiveProduct> findByRoomId(Long roomId);
    List<LiveProduct> findByProductId(Long productId);
    List<LiveProduct> findAll();
    void deleteById(Long id);
}