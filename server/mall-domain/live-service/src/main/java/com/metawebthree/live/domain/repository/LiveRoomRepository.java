package com.metawebthree.live.domain.repository;

import com.metawebthree.live.domain.model.LiveRoom;

import java.util.List;

public interface LiveRoomRepository {
    LiveRoom save(LiveRoom room);
    LiveRoom findById(Long id);
    List<LiveRoom> findByAnchorId(Long anchorId);
    List<LiveRoom> findByStatus(Integer status);
    List<LiveRoom> findAll();
    void deleteById(Long id);
}