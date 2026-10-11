package com.metawebthree.live.domain.repository;

import com.metawebthree.live.domain.model.LiveComment;

import java.util.List;

public interface LiveCommentRepository {
    LiveComment save(LiveComment comment);
    LiveComment findById(Long id);
    List<LiveComment> findByRoomId(Long roomId);
    List<LiveComment> findByUserId(Long userId);
    List<LiveComment> findAll();
    void deleteById(Long id);
}